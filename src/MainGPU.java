import boilerplate.common.GameBase;
import boilerplate.common.TimeStepper;
import boilerplate.common.Window;
import boilerplate.rendering.Renderer;
import boilerplate.rendering.ShaderProgram;
import boilerplate.rendering.buffers.ShaderStorageBuffer;
import boilerplate.rendering.camera.Camera;
import boilerplate.rendering.camera.CameraOrtho;
import boilerplate.utility.Logging;
import org.ejml.simple.SimpleMatrix;
import org.joml.Vector2f;
import org.lwjgl.glfw.GLFW;
import org.lwjgl.opengl.GL45;
import org.lwjgl.system.MemoryUtil;

import java.awt.*;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Random;

public class MainGPU extends GameBase {
    Window window = new Window();
    Camera camera = new CameraOrtho(new Vector2f());

    ShaderProgram runShader = new ShaderProgram();
    ShaderProgram trainingShader = new ShaderProgram();
    ShaderProgram learningShader = new ShaderProgram();

    static int trainingDataCount = 10;
    static int[] layerNodes = new int[]{2, 3, 3, 1};

    static float[] weights;
    static float[] biases;

    static float[] inputData = inputData();  // n0 by m
    static SimpleMatrix outputLabels = outputLabels();  // n3 by m

    public static void main(String[] args) {
        System.setProperty("joml.format", "false");
        new MainGPU().start();
    }

    @Override
    public void start() {
        TimeStepper.startStaticTimeStepper(1d / 60d, this);
    }

    @Override
    public void createCapabilitiesAndOpen() {
        Window.Options winOps = new Window.Options();
        winOps.title = "uwaaaaaa";
        winOps.initWindowSize = new Dimension(200, 200);
        window.quickSetupAndShow(winOps);
        camera.setupUniformBuffer();
        bindEvents();

        runShader.genProgram();
        runShader.attachShader("res/run.glsl", GL45.GL_COMPUTE_SHADER);
        runShader.linkProgram();
        trainingShader.genProgram();
        trainingShader.attachShader("res/train.glsl", GL45.GL_COMPUTE_SHADER);
        trainingShader.linkProgram();
        learningShader.genProgram();
        learningShader.attachShader("res/learn.glsl", GL45.GL_COMPUTE_SHADER);
        learningShader.linkProgram();

        runNN();
        trainNN();
    }

    public void bindEvents() {
        GL45.glDebugMessageCallback(Logging.debugCallback(), -1);

        GLFW.glfwSetKeyCallback(window.handle, (window, key, scancode, action, mods) -> {
            if (action == GLFW.GLFW_PRESS) {
                if (key == GLFW.GLFW_KEY_ESCAPE) this.window.setToClose();
            }
        });
    }

    // inputs: weight, height
    public static float[] inputData() {
        float[] out = new float[] {
                150, 70,
                254, 73,
                312, 68,
                120, 60,
                154, 61,
                212, 65,
                216, 67,
                145, 67,
                184, 64,
                130, 69
        };

        // find mean & deviation
        double meanWeight = 0;
        double meanHeight = 0;
        for (int i = 0; i < trainingDataCount; i++) {
            meanWeight += out[i * 2];
            meanHeight += out[i * 2 + 1];
        }
        meanWeight /= trainingDataCount;
        meanHeight /= trainingDataCount;

        double deviationWeight = 0;
        double deviationHeight = 0;
        for (int i = 0; i < trainingDataCount; i++) {
            deviationWeight += Math.pow(out[i * 2] - meanWeight, 2);
            deviationHeight += Math.pow(out[i * 2 + 1] - meanHeight, 2);
        }
        deviationWeight = Math.sqrt(deviationWeight / trainingDataCount);
        deviationHeight = Math.sqrt(deviationHeight / trainingDataCount);

        // apply standard scaling
        for (int i = 0; i < trainingDataCount; i++) {
            out[i * 2] = (float) ((out[i * 2] - meanWeight) / deviationWeight);
            out[i * 2 + 1] = (float) ((out[i * 2 + 1] - meanHeight) / deviationHeight);
        }
        return out;
    }

    public static SimpleMatrix outputLabels() {
        SimpleMatrix out = SimpleMatrix.ones(layerNodes[layerNodes.length-1], trainingDataCount);
        out.set(0, 0);
        out.set(1, 1);
        out.set(2, 1);
        out.set(3, 0);
        out.set(4, 0);
        out.set(5, 1);
        out.set(6, 1);
        out.set(7, 0);
        out.set(8, 1);
        out.set(9, 0);
        return out;
    }

    public static double calcCost(SimpleMatrix yHat) {
        double summedLosses = 0;
        for (int i = 0; i < trainingDataCount; i++) {
            double yI = outputLabels.get(i);
            double yHatI = yHat.get(i);
            double loss = yI == 0 ? 1 - yHatI : yHatI;
            summedLosses -= Math.log(loss);
        }
        return (1d / trainingDataCount) * summedLosses;
    }

    public static void randomizeWeightsAndBiases(int weightsCount, int biasesCount) {
        weights = new float[weightsCount];
        biases = new float[biasesCount];
        for (int i = 0; i < weightsCount; i++) weights[i] = (float) new Random().nextGaussian();
        for (int i = 0; i < biasesCount; i++) biases[i] = (float) new Random().nextGaussian();
    }

    public void runNN() {
        runShader.bind();

        // send layer data
        runShader.uniform1i("layers[0].size", layerNodes[0]);
        int weightOffset = 0;
        int biasesOffset = 0;
        for (int l = 1; l < layerNodes.length; l++) {
            runShader.uniform1i("layers[%s].size".formatted(l), layerNodes[l]);
            runShader.uniform1i("layers[%s].weightsOffset".formatted(l), weightOffset);
            runShader.uniform1i("layers[%s].biasesOffset".formatted(l), biasesOffset);
            weightOffset += layerNodes[l-1] * layerNodes[l];
            biasesOffset += layerNodes[l];
        }

        ShaderStorageBuffer ssbInputs = new ShaderStorageBuffer(true);
        ssbInputs.bindShaderToBlock(0, runShader);
        ssbInputs.bufferData(inputData);

        ShaderStorageBuffer ssbWeights = new ShaderStorageBuffer(true);
        ssbWeights.bindShaderToBlock(1, runShader);

        ShaderStorageBuffer ssbBiases = new ShaderStorageBuffer(true);
        ssbBiases.bindShaderToBlock(2, runShader);

        ShaderStorageBuffer ssbOutput = new ShaderStorageBuffer(true);
        ssbOutput.bindShaderToBlock(3, runShader);
        ssbOutput.bufferSize(layerNodes[layerNodes.length-1] * trainingDataCount * Float.BYTES);

        randomizeWeightsAndBiases(weightOffset, biasesOffset);
        ssbWeights.bufferData(weights);
        ssbBiases.bufferData(biases);

        GL45.glDispatchCompute(trainingDataCount, 1, 1);
        GL45.glMemoryBarrier(GL45.GL_ALL_BARRIER_BITS);

        ssbOutput.bind();
        ByteBuffer output = MemoryUtil.memAlloc(layerNodes[layerNodes.length - 1] * trainingDataCount * Float.BYTES);
        GL45.glGetBufferSubData(GL45.GL_SHADER_STORAGE_BUFFER, 0, output);
        for (int i = 0; i < trainingDataCount; i++) System.out.println(output.asFloatBuffer().get(i));
    }

    public void trainNN() {
        trainingShader.bind();
    }

    public void learnNN() {
        learningShader.bind();
    }

    @Override
    public void mainLoop(double v) {
        GLFW.glfwPollEvents();
        Renderer.clearCDS();
        Renderer.finish(window);
    }

    @Override
    public boolean shouldClose() {
        return GLFW.glfwWindowShouldClose(window.handle);
    }

    @Override
    public void close() {
        window.close();
    }
}

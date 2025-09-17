import boilerplate.common.GameBase;
import boilerplate.common.TimeStepper;
import boilerplate.common.Window;
import boilerplate.rendering.Renderer;
import boilerplate.rendering.ShaderProgram;
import boilerplate.rendering.buffers.ShaderStorageBuffer;
import boilerplate.rendering.buffers.UniformBuffer;
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
import java.nio.FloatBuffer;
import java.util.Random;

public class MainGPU extends GameBase {
    Window window = new Window();
    Camera camera = new CameraOrtho(new Vector2f());

    ShaderProgram runShader = new ShaderProgram();
    ShaderStorageBuffer[] runShaderInputs = new ShaderStorageBuffer[5];
    ShaderProgram backpropShader = new ShaderProgram();
    ShaderStorageBuffer[] backpropShaderInputs = new ShaderStorageBuffer[4];
    ShaderProgram learnShader = new ShaderProgram();

    static int trainingDataCount = 10;
    static int[] layerNodes = new int[]{2, 3, 3, 1};

    static float[] weights;
    static float[] biases;

    static float[] inputData = inputData();  // n0 by m
    static float[] outputLabels = new float[] {0, 1, 1, 0, 0, 1, 1, 0, 1, 0};  // n3 by m

    public static void main(String[] args) {
        System.setProperty("joml.format", "false");
        new MainGPU().start();
    }

    @Override
    public void start() {
        TimeStepper.startStaticTimeStepper(1d / 60d, this);
    }

    public int[] uniformLayers(ShaderProgram sh) {
        sh.uniform1i("layers[0].size", layerNodes[0]);
        int weightOffset = 0;
        int biasesOffset = 0;
        for (int l = 1; l < layerNodes.length; l++) {
            sh.uniform1i("layers[%s].size".formatted(l), layerNodes[l]);
            sh.uniform1i("layers[%s].weightsOffset".formatted(l), weightOffset);
            sh.uniform1i("layers[%s].biasesOffset".formatted(l), biasesOffset);
            weightOffset += layerNodes[l-1] * layerNodes[l];
            biasesOffset += layerNodes[l];
        }
        return new int[] {weightOffset, biasesOffset};
    }

    public void initialiseStorageBuffers(ShaderStorageBuffer[] buffers, ShaderProgram sh) {
        for (int i = 0; i < buffers.length; i++) {
            buffers[i] = new ShaderStorageBuffer(true);
            buffers[i].bindShaderToBlock(i, sh);
        }
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
        backpropShader.genProgram();
        backpropShader.attachShader("res/backprop2.glsl", GL45.GL_COMPUTE_SHADER);
        backpropShader.linkProgram();
//        learnShader.genProgram();
//        learnShader.attachShader("res/learn.glsl", GL45.GL_COMPUTE_SHADER);
//        learnShader.linkProgram();

        int[] sizes = uniformLayers(runShader);
        uniformLayers(backpropShader);
        randomizeWeightsAndBiases(sizes[0], sizes[1]);

        initialiseStorageBuffers(runShaderInputs, runShader);
        initialiseStorageBuffers(backpropShaderInputs, backpropShader);

        FloatBuffer out = runNN();
        for (int i = 0; i < trainingDataCount; i++) System.out.println(out.get(i));
//        trainNN();
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

//    public static double calcCost(SimpleMatrix yHat) {
//        double summedLosses = 0;
//        for (int i = 0; i < trainingDataCount; i++) {
//            double yI = outputLabels.get(i);
//            double yHatI = yHat.get(i);
//            double loss = yI == 0 ? 1 - yHatI : yHatI;
//            summedLosses -= Math.log(loss);
//        }
//        return (1d / trainingDataCount) * summedLosses;
//    }

    public static void randomizeWeightsAndBiases(int weightsCount, int biasesCount) {
        weights = new float[weightsCount];
        biases = new float[biasesCount];
        for (int i = 0; i < weightsCount; i++) weights[i] = (float) new Random().nextGaussian();
        for (int i = 0; i < biasesCount; i++) biases[i] = (float) new Random().nextGaussian();
    }

    public FloatBuffer runNN() {
        runShader.bind();

        int outputSize = layerNodes[layerNodes.length-1] * trainingDataCount * Float.BYTES;
        runShaderInputs[0].bufferData(inputData);
        runShaderInputs[1].bufferData(weights);
        runShaderInputs[2].bufferData(biases);
        runShaderInputs[3].bufferSize(outputSize);
        runShaderInputs[4].bufferSize(outputSize);

        GL45.glDispatchCompute(trainingDataCount, 1, 1);
        GL45.glMemoryBarrier(GL45.GL_ALL_BARRIER_BITS);

        runShaderInputs[3].bind();
        ByteBuffer output = MemoryUtil.memAlloc(outputSize);
        GL45.glGetBufferSubData(GL45.GL_SHADER_STORAGE_BUFFER, 0, output);
        return output.asFloatBuffer();
    }

    public void trainNN() {
        FloatBuffer outputs = runNN();
        backpropShader.bind();
        backpropShaderInputs[0].bufferData(outputs);
        backpropShaderInputs[1].bufferData(outputLabels);
        backpropShaderInputs[2].bufferData(weights);
        backpropShaderInputs[3].bufferData(biases);
    }

    public void learnNN() {
        learnShader.bind();
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

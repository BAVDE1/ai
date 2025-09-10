import boilerplate.common.GameBase;
import boilerplate.common.TimeStepper;
import boilerplate.common.Window;
import boilerplate.rendering.Renderer;
import boilerplate.rendering.ShaderProgram;
import boilerplate.rendering.camera.Camera;
import boilerplate.rendering.camera.CameraOrtho;
import boilerplate.utility.Logging;
import org.joml.Vector2f;
import org.lwjgl.glfw.GLFW;
import org.lwjgl.opengl.GL45;

import java.awt.*;

public class MainGPU extends GameBase {
    Window window = new Window();
    Camera camera = new CameraOrtho(new Vector2f(1));

    ShaderProgram trainingShader = new ShaderProgram();

    public static void main(String[] args) {
        System.setProperty("joml.format", "false");
//        Logging.logDebug = false;
        new MainGPU().start();
    }

    @Override
    public void start() {
        TimeStepper.startStaticTimeStepper(1d / 60d, this);
    }

    @Override
    public void createCapabilitiesAndOpen() {
        Window.Options winOps = new Window.Options();
        winOps.title = "the example index";
        winOps.initWindowSize = new Dimension(100, 100);
        window.quickSetupAndShow(winOps);
        camera.setupUniformBuffer();
        bindEvents();

        trainingShader.genProgram();
        trainingShader.attachShader("res/train.glsl", GL45.GL_COMPUTE_SHADER);
        trainingShader.linkProgram();
    }

    public void bindEvents() {
        GL45.glDebugMessageCallback(Logging.debugCallback(), -1);

        GLFW.glfwSetKeyCallback(window.handle, (window, key, scancode, action, mods) -> {
            if (action == GLFW.GLFW_PRESS) {
                if (key == GLFW.GLFW_KEY_ESCAPE) this.window.setToClose();
            }
        });
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

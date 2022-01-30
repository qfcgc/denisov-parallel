package dev.vlde;

import java.awt.BorderLayout;
import java.awt.Graphics;
import java.awt.GridLayout;
import java.awt.RenderingHints;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.awt.image.BufferedImageOp;
import java.awt.image.ColorModel;
import java.awt.image.ConvolveOp;
import java.awt.image.DataBufferInt;
import java.awt.image.Kernel;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_platform_id;
import org.jocl.cl_program;


import static org.jocl.CL.CL_CONTEXT_PLATFORM;
import static org.jocl.CL.CL_DEVICE_TYPE_ALL;
import static org.jocl.CL.CL_MEM_READ_ONLY;
import static org.jocl.CL.CL_MEM_USE_HOST_PTR;
import static org.jocl.CL.CL_MEM_WRITE_ONLY;
import static org.jocl.CL.CL_TRUE;
import static org.jocl.CL.clBuildProgram;
import static org.jocl.CL.clCreateBuffer;
import static org.jocl.CL.clCreateCommandQueue;
import static org.jocl.CL.clCreateContext;
import static org.jocl.CL.clCreateKernel;
import static org.jocl.CL.clCreateProgramWithSource;
import static org.jocl.CL.clEnqueueNDRangeKernel;
import static org.jocl.CL.clEnqueueReadBuffer;
import static org.jocl.CL.clEnqueueWriteBuffer;
import static org.jocl.CL.clGetDeviceIDs;
import static org.jocl.CL.clGetPlatformIDs;
import static org.jocl.CL.clReleaseCommandQueue;
import static org.jocl.CL.clReleaseContext;
import static org.jocl.CL.clReleaseKernel;
import static org.jocl.CL.clReleaseMemObject;
import static org.jocl.CL.clReleaseProgram;
import static org.jocl.CL.clSetKernelArg;
import static org.jocl.CL.setExceptionsEnabled;

public class ImageBlur {
    private final JLabel javaTimeLabel;
    private final JLabel joclTimeLabel;
    private final BufferedImage inputImage;
    private BufferedImage outputImageJava;
    private BufferedImage outputImageOpenCL;
    private List<Kernel> kernels;
    private List<String> kernelNames;

    public ImageBlur() {
        String fileName = "src/main/resources/sunset.jpg";

        inputImage = createBufferedImage(fileName);
        int sizeX = inputImage.getWidth();
        int sizeY = inputImage.getHeight();

        outputImageJava = new BufferedImage(
                sizeX, sizeY, BufferedImage.TYPE_INT_RGB);
        outputImageOpenCL = new BufferedImage(
                sizeX, sizeY, BufferedImage.TYPE_INT_RGB);

        initKernels();

        final JPanel mainPanel = new JPanel(new GridLayout(1, 0));
        JPanel panel;

        final JComboBox<String> kernelComboBox =
                new JComboBox<>(kernelNames.toArray(new String[0]));
        kernelComboBox.addActionListener(e -> {
            int index = kernelComboBox.getSelectedIndex();
            Kernel kernel = kernels.get(index);
            applyKernel(kernel);
            mainPanel.repaint();
        });
        panel = new JPanel(new BorderLayout());
        panel.add(new JLabel(new ImageIcon(inputImage)), BorderLayout.CENTER);
        panel.add(kernelComboBox, BorderLayout.NORTH);
        mainPanel.add(panel);

        javaTimeLabel = new JLabel();
        javaTimeLabel.setPreferredSize(kernelComboBox.getPreferredSize());
        panel = new JPanel(new BorderLayout());
        panel.add(new JLabel(new ImageIcon(outputImageJava)), BorderLayout.CENTER);
        panel.add(javaTimeLabel, BorderLayout.NORTH);
        mainPanel.add(panel);

        joclTimeLabel = new JLabel();
        joclTimeLabel.setPreferredSize(kernelComboBox.getPreferredSize());
        panel = new JPanel(new BorderLayout());
        panel.add(new JLabel(new ImageIcon(outputImageOpenCL)), BorderLayout.CENTER);
        panel.add(joclTimeLabel, BorderLayout.NORTH);
        mainPanel.add(panel);

        JFrame frame = new JFrame("Blur");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLayout(new BorderLayout());
        frame.add(mainPanel, BorderLayout.CENTER);
        frame.pack();
        frame.setVisible(true);

        kernelComboBox.setSelectedIndex(0);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(ImageBlur::new);
    }

    private static BufferedImage createBufferedImage(String fileName) {
        BufferedImage image;
        try {
            image = ImageIO.read(new File(fileName));
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }

        int sizeX = image.getWidth();
        int sizeY = image.getHeight();

        BufferedImage result = new BufferedImage(
                sizeX, sizeY, BufferedImage.TYPE_INT_RGB);
        Graphics g = result.createGraphics();
        g.drawImage(image, 0, 0, null);
        g.dispose();
        return result;
    }

    private void applyKernel(Kernel kernel) {
        long before;
        long after;
        double durationMS;
        String message;

        BufferedImageOp bop = new ConvolveOp(kernel);
        before = System.nanoTime();
        outputImageJava = bop.filter(inputImage, outputImageJava);
        after = System.nanoTime();
        durationMS = (after - before) / 1e6;
        message = "Java: " + String.format("%.2f", durationMS) + " ms";
        System.out.println(message);
        javaTimeLabel.setText(message);

        JoclBlurOp jop = JoclBlurOp.create(kernel);
        before = System.nanoTime();
        outputImageOpenCL = jop.filter(inputImage, outputImageOpenCL);
        after = System.nanoTime();
        durationMS = (after - before) / 1e6;
        message = "JOCL: " + String.format("%.2f", durationMS) + " ms";
        System.out.println(message);
        joclTimeLabel.setText(message);
        jop.shutdown();
    }

    private void initKernels() {
        kernels = new ArrayList<>();
        kernelNames = new ArrayList<>();
        // Blur
        for (int i = 3; i <= 15; i += 2) {
            initBlurKernel(i);
        }
    }

    private void initBlurKernel(int kernelSize) {
        int size = kernelSize * kernelSize;
        float value = 1.0f / size;
        float[] kernelData = new float[size];
        for (int i = 0; i < size; i++) {
            kernelData[i] = value;
        }
        kernels.add(new Kernel(kernelSize, kernelSize, kernelData));
        kernelNames.add("Blur " + kernelSize + "x" + kernelSize);
    }
}

class JoclBlurOp implements BufferedImageOp {
    private static final String KERNEL_SOURCE_FILE_NAME = "src/main/resources/blur.cl";
    private final cl_context context;
    private final cl_command_queue commandQueue;
    private final cl_kernel clKernel;
    private final Kernel kernel;
    private final cl_mem kernelMem;

    public JoclBlurOp(
            cl_context context, cl_command_queue commandQueue, Kernel kernel) {
        this.context = context;
        this.commandQueue = commandQueue;
        this.kernel = kernel;

        String source = readFile(KERNEL_SOURCE_FILE_NAME);
        cl_program program = clCreateProgramWithSource(context, 1,
                new String[]{source}, null, null);
        String compileOptions = "-cl-mad-enable";
        clBuildProgram(program, 0, null, compileOptions, null, null);
        clKernel = clCreateKernel(program, "blur", null);
        clReleaseProgram(program);

        float[] kernelData = kernel.getKernelData(null);
        kernelMem = clCreateBuffer(context, CL_MEM_READ_ONLY,
                (long) kernelData.length * Sizeof.cl_uint, null, null);
        clEnqueueWriteBuffer(commandQueue, kernelMem,
                true, 0, (long) kernelData.length * Sizeof.cl_uint,
                Pointer.to(kernelData), 0, null, null);
    }

    private static long round(long groupSize, long globalSize) {
        long r = globalSize % groupSize;
        if (r == 0) {
            return globalSize;
        } else {
            return globalSize + groupSize - r;
        }
    }

    private static String readFile(String fileName) {
        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(fileName));
            StringBuilder sb = new StringBuilder();
            String line;
            while (true) {
                line = br.readLine();
                if (line == null) {
                    break;
                }
                sb.append(line).append("\n");
            }
            return sb.toString();
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
            return "";
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public static JoclBlurOp create(Kernel kernel) {
        final int platformIndex = 0;
        final int deviceIndex = 1;
        final long deviceType = CL_DEVICE_TYPE_ALL;

        setExceptionsEnabled(true);

        int[] numPlatformsArray = new int[1];
        clGetPlatformIDs(0, null, numPlatformsArray);
        int numPlatforms = numPlatformsArray[0];

        cl_platform_id[] platforms = new cl_platform_id[numPlatforms];
        clGetPlatformIDs(platforms.length, platforms, null);
        cl_platform_id platform = platforms[platformIndex];

        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);

        int[] numDevicesArray = new int[1];
        clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
        int numDevices = numDevicesArray[0];

        cl_device_id[] devices = new cl_device_id[numDevices];
        clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
        cl_device_id device = devices[deviceIndex];

        cl_context context = clCreateContext(
                contextProperties, 1, new cl_device_id[]{device},
                null, null, null);

        cl_command_queue commandQueue = clCreateCommandQueue(context, device, 0, null);

        return new JoclBlurOp(context, commandQueue, kernel);
    }

    public void shutdown() {
        clReleaseMemObject(kernelMem);
        clReleaseKernel(clKernel);
        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);
    }


    @Override
    public BufferedImage createCompatibleDestImage(
            BufferedImage src, ColorModel destCM) {
        int w = src.getWidth();
        int h = src.getHeight();
        return new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
    }

    @Override
    public BufferedImage filter(BufferedImage src, BufferedImage dst) {
        if (src.getType() != BufferedImage.TYPE_INT_RGB) {
            throw new IllegalArgumentException("Source image is not TYPE_INT_RGB");
        }
        if (dst == null) {
            dst = createCompatibleDestImage(src, null);
        } else if (dst.getType() != BufferedImage.TYPE_INT_RGB) {
            throw new IllegalArgumentException("Destination image is not TYPE_INT_RGB");
        }
        if (src.getWidth() != dst.getWidth() ||
                src.getHeight() != dst.getHeight()) {
            throw new IllegalArgumentException("Images do not have the same size");
        }
        int imageSizeX = src.getWidth();
        int imageSizeY = src.getHeight();

        DataBufferInt dataBufferSrc =
                (DataBufferInt) src.getRaster().getDataBuffer();
        int[] dataSrc = dataBufferSrc.getData();
        cl_mem inputImageMem = clCreateBuffer(context,
                CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                (long) dataSrc.length * Sizeof.cl_uint,
                Pointer.to(dataSrc), null);

        cl_mem outputImageMem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                (long) imageSizeX * imageSizeY * Sizeof.cl_uint, null, null);

        int kernelSizeX = kernel.getWidth();
        int kernelSizeY = kernel.getHeight();
        int kernelOriginX = kernel.getXOrigin();
        int kernelOriginY = kernel.getYOrigin();

        long[] localWorkSize = new long[2];
        localWorkSize[0] = kernelSizeX;
        localWorkSize[1] = kernelSizeY;

        long[] globalWorkSize = new long[2];
        globalWorkSize[0] = round(localWorkSize[0], imageSizeX);
        globalWorkSize[1] = round(localWorkSize[1], imageSizeY);

        int[] imageSize = new int[]{imageSizeX, imageSizeY};
        int[] kernelSize = new int[]{kernelSizeX, kernelSizeY};
        int[] kernelOrigin = new int[]{kernelOriginX, kernelOriginY};

        clSetKernelArg(clKernel, 0, Sizeof.cl_mem, Pointer.to(inputImageMem));
        clSetKernelArg(clKernel, 1, Sizeof.cl_mem, Pointer.to(kernelMem));
        clSetKernelArg(clKernel, 2, Sizeof.cl_mem, Pointer.to(outputImageMem));
        clSetKernelArg(clKernel, 3, Sizeof.cl_int2, Pointer.to(imageSize));
        clSetKernelArg(clKernel, 4, Sizeof.cl_int2, Pointer.to(kernelSize));
        clSetKernelArg(clKernel, 5, Sizeof.cl_int2, Pointer.to(kernelOrigin));

        clEnqueueNDRangeKernel(commandQueue, clKernel, 2, null,
                globalWorkSize, localWorkSize, 0, null, null);

        DataBufferInt dataBufferDst =
                (DataBufferInt) dst.getRaster().getDataBuffer();
        int[] dataDst = dataBufferDst.getData();
        clEnqueueReadBuffer(commandQueue, outputImageMem,
                CL_TRUE, 0, (long) dataDst.length * Sizeof.cl_uint,
                Pointer.to(dataDst), 0, null, null);

        clReleaseMemObject(inputImageMem);
        clReleaseMemObject(outputImageMem);

        return dst;
    }

    @Override
    public Rectangle2D getBounds2D(BufferedImage src) {
        return src.getRaster().getBounds();
    }

    @Override
    public final Point2D getPoint2D(Point2D srcPt, Point2D dstPt) {
        if (dstPt == null) {
            dstPt = new Point2D.Float();
        }
        dstPt.setLocation(srcPt.getX(), srcPt.getY());
        return dstPt;
    }

    @Override
    public RenderingHints getRenderingHints() {
        return null;
    }
}

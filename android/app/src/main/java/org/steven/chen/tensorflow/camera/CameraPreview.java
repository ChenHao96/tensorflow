package org.steven.chen.tensorflow.camera;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.ImageFormat;
import android.hardware.Camera;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

import java.io.IOException;
import java.util.List;

@SuppressLint("ViewConstructor")
public class CameraPreview extends SurfaceView implements SurfaceHolder.Callback {

    private static final String TAG = "CameraPreview";

    private Camera mCamera;
    private SurfaceHolder mHolder;

    public CameraPreview(Context context, Camera camera) {
        super(context);
        this.mCamera = camera;

        this.mHolder = getHolder();
        this.mHolder.addCallback(this);
        this.mHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        initCameraParams(holder, "Error setting camera preview: ");
    }

    private Camera.Size getBastSize(List<Camera.Size> sizes) {
        if (sizes == null || sizes.size() < 1) return null;
        Camera.Size result = null;
        for (Camera.Size size : sizes) {
            if (result == null) {
                result = size;
            } else {
                if (size.width - result.width > 0) {
                    result = size;
                }
            }
        }
        return result;
    }

    private void initCameraParams(SurfaceHolder holder, String message) {
        Camera.Parameters parameters = mCamera.getParameters();
        parameters.setPictureFormat(ImageFormat.JPEG);
        parameters.setJpegQuality(100);

        Camera.Size pictureSize = getBastSize(parameters.getSupportedPictureSizes());
        if (pictureSize != null) {
            parameters.setPictureSize(pictureSize.width, pictureSize.height);
        }

        Camera.Size previewSize = getBastSize(parameters.getSupportedPreviewSizes());
        if (previewSize != null) {
            parameters.setPreviewSize(previewSize.width, previewSize.height);
        }

        try {
            mCamera.addCallbackBuffer(new byte[((getWidth() * getHeight()) *
                    ImageFormat.getBitsPerPixel(ImageFormat.NV21)) / 8]);

            mCamera.setDisplayOrientation(90);
            mCamera.setParameters(parameters);
            mCamera.setPreviewDisplay(holder);
            mCamera.startPreview();
        } catch (IOException e) {
            Log.d(TAG, message + e.getMessage());
        }
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
        // empty. Take care of releasing the Camera preview in your activity.
        this.mCamera = null;
        this.mHolder = null;
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int w, int h) {
        // If your preview can change or rotate, take care of those events here.
        // Make sure to stop the preview before resizing or reformatting it.

        if (mHolder.getSurface() == null) {
            // preview surface does not exist
            return;
        }

        // stop preview before making changes
        try {
            mCamera.stopPreview();
        } catch (Exception e) {
            // ignore: tried to stop a non-existent preview
        }

        // set preview size and make any resize, rotate or
        // reformatting changes here

        // start preview with new settings
        initCameraParams(mHolder, "Error starting camera preview: ");
    }
}
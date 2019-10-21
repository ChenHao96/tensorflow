package org.steven.chen.tensorflow.camera;

import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Point;
import android.graphics.Rect;
import android.hardware.Camera;
import android.util.Log;
import android.view.View;

import java.util.ArrayList;
import java.util.List;

public class CameraUtil {

    private static final String TAG = "CameraUtil";
    private static final int CAMERA_FOCUS_RADIUS = 100;

    public static final int SUCCESS_CODE = 0;
    public static final int EXCEPTION_CODE = -2;
    public static final int PARAM_FAIL_CODE = -1;
    public static final int CAMERA_FOCUS_OUT_OF_CODE = 1;
    public static final int CAMERA_METERING_OUT_OF_CODE = 2;

    private static boolean checkCameraHardware(Context context) {
        return context.getPackageManager().hasSystemFeature(PackageManager.FEATURE_CAMERA);
    }

    public static Camera getCameraInstance(Context context, int cameraId) {
        if (checkCameraHardware(context)) {
            try {
                return Camera.open(cameraId);
            } catch (Exception e) {
                Log.d(TAG, "Error open camera: " + e.getMessage());
            }
        }
        Log.d(TAG, "No camera on this device");
        return null;
    }

    public static void releaseCamera(Camera camera) {
        if (camera != null) {
            camera.stopPreview();
            camera.setAutoFocusMoveCallback(null);
            camera.setPreviewCallback(null);
            camera.release();
        }
    }

    public static boolean isSupportZoom(Camera camera) {
        if (camera == null) return false;
        return camera.getParameters().isSmoothZoomSupported();
    }

    public static List<Camera.Area> getUserClickPoint2CameraArea(View view, int radius, Point... points) {
        if (view == null) return null;
        if (points == null || points.length == 0) return null;
        List<Camera.Area> result = new ArrayList<>(points.length);
        for (Point point : points) {
            if (point == null) continue;
            Rect focusArea = new Rect();

            int areaX = (int) (((double) point.x / (double) view.getWidth()) * 2000 - 1000);
            int areaY = (int) (((double) point.y / (double) view.getHeight()) * 2000 - 1000);
            focusArea.left = Math.max(areaX - radius, -1000);
            focusArea.top = Math.max(areaY - radius, -1000);
            focusArea.right = Math.min(areaX + radius, 1000);
            focusArea.bottom = Math.min(areaY + radius, 1000);
            result.add(new Camera.Area(focusArea, 1000));
            Log.i(TAG, String.format("focusArea(left:%d,top:%d,right:%d,bottom:%d)",
                    focusArea.left, focusArea.top, focusArea.right, focusArea.bottom));
        }
        return result;
    }

    public static void updateCameraZoom(Camera camera, int zoomUpdate) {
        Camera.Parameters params = camera.getParameters();
        if (params.isZoomSupported()) {
            int zoom = params.getZoom();
            zoom += zoomUpdate;
            zoom = Math.min(zoom, params.getMaxZoom());
            zoom = Math.max(zoom, 0);
            params.setZoom(zoom);
            camera.setParameters(params);
        }
    }

    public static int userClickFocus(View view, Camera camera, Point... points) {
        if (points == null || points.length == 0) {
            Log.d(TAG, "points param fail");
            return PARAM_FAIL_CODE;
        }

        int result = SUCCESS_CODE;
        Camera.Parameters mParameters = camera.getParameters();
        mParameters.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_PICTURE);

        //TODO:存在对焦问题
        List<Camera.Area> cameraAreas = getUserClickPoint2CameraArea(view, CAMERA_FOCUS_RADIUS, points);
        if (points.length <= mParameters.getMaxNumMeteringAreas()) {
            mParameters.setMeteringAreas(cameraAreas);
        } else {
            Log.d(TAG, "points more than the metering areas");
            result = CAMERA_METERING_OUT_OF_CODE;
        }

        if (points.length <= mParameters.getMaxNumFocusAreas()) {
            mParameters.setFocusAreas(cameraAreas);
        } else {
            Log.d(TAG, "points more than the focus areas");
            result = CAMERA_FOCUS_OUT_OF_CODE;
        }

        try {
            camera.cancelAutoFocus();
            camera.setParameters(mParameters);
        } catch (Exception e) {
            Log.w(TAG, "set camera parameters fail", e);
            result = EXCEPTION_CODE;
        }
        return result;
    }
}

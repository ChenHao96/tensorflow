package org.steven.chen.tensorflow.camera;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.Point;
import android.graphics.PorterDuff;
import android.graphics.Rect;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

import org.steven.chen.tensorflow.Commons;

import java.util.List;

public class UserFocusPreview extends SurfaceView {

    private static final String TAG = "UserFocusPreview";

    private Paint paint;
    private SurfaceHolder mHolder;
    private boolean hasDraw = false;

    public static final int RADIUS = 100;

    public UserFocusPreview(Context context) {
        super(context);
        super.setZOrderOnTop(true);

        this.mHolder = super.getHolder();
        this.mHolder.setFormat(PixelFormat.TRANSPARENT);
        this.mHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);

        this.paint = new Paint();
        this.paint.setStrokeWidth(1);
        this.paint.setColor(Color.WHITE);
    }

    public void setUserFocusArea(Point point) {
        if (point == null) return;
        this.setUserFocusArea(Commons.clickPoint2Rect(this, RADIUS, point));
    }

    public void clearFocusArea() {
        if (this.hasDraw) {
            Canvas canvas = this.mHolder.lockCanvas(new Rect(0, 0, getWidth(), getHeight()));
            canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);
            this.mHolder.unlockCanvasAndPost(canvas);
            this.hasDraw = false;
        }
    }

    public void setUserFocusArea(List<Rect> areas) {
        if (areas == null || areas.size() == 0) return;
        Canvas canvas = this.mHolder.lockCanvas(new Rect(0, 0, getWidth(), getHeight()));
        canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);
        try {
            for (Rect area : areas) {
                if (area == null) continue;
                canvas.drawLine(area.left, area.top, area.left, area.bottom, paint);
                canvas.drawLine(area.right, area.top, area.right, area.bottom, paint);
                canvas.drawLine(area.left, area.top, area.right, area.top, paint);
                canvas.drawLine(area.left, area.bottom, area.right, area.bottom, paint);
                this.hasDraw = true;
            }
        } catch (Exception e) {
            Log.d(TAG, "draw user focus area fail", e);
        } finally {
            this.mHolder.unlockCanvasAndPost(canvas);
        }
    }
}

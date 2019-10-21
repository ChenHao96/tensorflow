package org.steven.chen.tensorflow;

import android.graphics.Point;
import android.graphics.Rect;
import android.view.View;

import java.util.ArrayList;
import java.util.List;

public class Commons {

    public static List<Rect> clickPoint2Rect(View view, int radius, Point... points) {
        if (view == null) return null;
        if (points == null || points.length == 0) return null;
        List<Rect> result = new ArrayList<>(points.length);
        for (Point point : points) {
            if (point == null) continue;
            Rect focusArea = new Rect();
            focusArea.left = Math.max(point.x - radius, 0);
            focusArea.top = Math.max(point.y - radius, 0);
            focusArea.right = Math.min(point.x + radius, view.getWidth());
            focusArea.bottom = Math.min(point.y + radius, view.getHeight());
            result.add(focusArea);
        }
        return result;
    }
}

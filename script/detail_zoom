import cv2
import math
import os.path
from PIL import Image, ImageDraw, ImageTk

def select_roi_region(image_path, line_color=(0, 255, 0), zoom_factor=3):
    """
    Interactive ROI selection tool with real-time zoom preview
    Args:
        image_path: Path to input image
        line_color: BGR color tuple for rectangle (default: green)
        zoom_factor: Multiplier for preview window zoom (default: 3x)
    Returns:
        tuple: (x, y, width, height) of selected region
    """
    drawing = False
    ix, iy = -1, -1
    x, y, w, h = 0, 0, 0, 0
    img = cv2.imread(image_path)
    clone = img.copy()

    # 鼠标回调函数
    def mouse_callback(event, cur_x, cur_y, flags, param):
        nonlocal ix, iy, drawing, x, y, w, h

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = cur_x, cur_y
            x, y, w, h = 0, 0, 0, 0

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            temp_img = clone.copy()
            cv2.rectangle(temp_img, (ix, iy), (cur_x, cur_y), line_color, 2)
            x1, y1 = min(ix, cur_x), min(iy, cur_y)
            x2, y2 = max(ix, cur_x), max(iy, cur_y)

            if x2 > x1 and y2 > y1:
                try:
                    roi = img[y1:y2, x1:x2]
                    if roi.size > 0:
                        enlarged = cv2.resize(roi, None, fx=3, fy=3,
                                              interpolation=cv2.INTER_CUBIC)
                        cv2.imshow("Enlarged Preview", enlarged)
                except Exception as e:
                    pass

            cur_w = abs(cur_x - ix)
            cur_h = abs(cur_y - iy)
            if cur_w > 0 and cur_h > 0:
                try:
                    roi = img[y1:y2, x1:x2]
                    enlarged = cv2.resize(roi, None, fx=zoom_factor, fy=zoom_factor,
                                          interpolation=cv2.INTER_CUBIC)
                    cv2.imshow("Enlarged Preview", enlarged)
                except:
                    pass
            cv2.putText(temp_img, f"X:{x1} Y:{y1} W:{cur_w} H:{cur_h}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2)
            cv2.imshow("Select ROI (SPACE=Clear | ENTER=Confirm)", temp_img)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            x = min(ix, cur_x)
            y = min(iy, cur_y)
            w = abs(cur_x - ix)
            h = abs(cur_y - iy)
            cv2.rectangle(clone, (x, y), (x + w, y + h), line_color, 2)
            cv2.putText(clone, f"X:{x} Y:{y} W:{w} H:{h}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2)
            cv2.imshow("Select ROI (SPACE=Clear | ENTER=Confirm)", clone)

    cv2.namedWindow("Select ROI (SPACE=Clear | ENTER=Confirm)")
    cv2.setMouseCallback("Select ROI (SPACE=Clear | ENTER=Confirm)", mouse_callback)

    while True:
        cv2.imshow("Select ROI (SPACE=Clear | ENTER=Confirm)", clone)
        key = cv2.waitKey(1) & 0xFF
        # 空格键：清除选择
        if key == 32:
            clone = img.copy()
            ix, iy = -1, -1
            x, y, w, h = 0, 0, 0, 0
            try:
                cv2.destroyWindow("Enlarged Preview") if cv2.getWindowProperty("Enlarged Preview", 0) >= 0 else None
            except:
                pass
            cv2.imshow("Select ROI (SPACE=Clear | ENTER=Confirm)", clone)
        # 回车键：确认选择
        if key == 13:
            try:
                cv2.destroyWindow("Enlarged Preview")
            except:
                pass
            break
    cv2.destroyAllWindows()
    return (x, y, w, h)

def plot_highlight_region(image_path, region_to_zoom, paste_position=None, zoom_factor=3,
                          line_color="red", line_wide=2, show_arrow=True, arrow_size=5):
    """
    Visualize image regions with zoomed preview and connection arrows.

    Args:
        image_path: Path to source image
        region_to_zoom: Tuple (x, y, w, h) defining region to highlight
        paste_position: Custom position (x,y) for zoomed preview (auto-placed if None)
        zoom_factor: Scaling multiplier for preview
        line_color: Border/arrow color name or hex code
        line_wide: Border/arrow line width
        show_arrow: Toggle connection arrow visibility
        arrow_size: Relative size multiplier for arrowhead

    Returns:
        PIL.Image: Composite image with visual annotations
   """
    x, y, w, h = region_to_zoom
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size
    original_copy = img.copy()
    zoomed_w = int(w * zoom_factor)
    zoomed_h = int(h * zoom_factor)
    cropped = original_copy.crop((x, y, x + w, y + h))
    zoomed = cropped.resize((zoomed_w, zoomed_h), Image.Resampling.LANCZOS)

    if paste_position is None:
        if x + w < img_w / 2:
            paste_x = img_w - zoomed_w
        else:
            paste_x = 0
        if y + h < img_h / 2:
            paste_y = img_h - zoomed_h
        else:
            paste_y = 0
    else:
        paste_x, paste_y = paste_position

    # 把线宽也考虑进去，保证缩放框的边框不会被裁掉
    margin = line_wide
    paste_x = max(margin, min(paste_x, img_w - zoomed_w - margin))
    paste_y = max(margin, min(paste_y, img_h - zoomed_h - margin))
    paste_position = (paste_x, paste_y)

    img.paste(zoomed, paste_position)
    draw = ImageDraw.Draw(img)
    draw.rectangle([(x, y), (x + w, y + h)],
                   outline=line_color,
                   width=line_wide)
    paste_x, paste_y = paste_position
    draw.rectangle([paste_position,
                    (paste_x + zoomed_w, paste_y + zoomed_h)],
                   outline=line_color, width=line_wide)
    if show_arrow:
        def get_side_center(rect, side):
            x, y, w, h = rect
            return {
                'left': (x, y + h // 2),
                'right': (x + w, y + h // 2),
                'top': (x + w // 2, y),
                'bottom': (x + w // 2, y + h)
            }[side]

        src_rect = (x, y, w, h)
        dst_rect = (paste_position[0], paste_position[1], zoomed_w, zoomed_h)
        dx = (dst_rect[0] + zoomed_w / 2) - (x + w / 2)
        dy = (dst_rect[1] + zoomed_h / 2) - (y + h / 2)
        if abs(dx) > abs(dy):
            src_side = 'right' if dx > 0 else 'left'
            dst_side = 'left' if dx > 0 else 'right'
        else:
            src_side = 'bottom' if dy > 0 else 'top'
            dst_side = 'top' if dy > 0 else 'bottom'

        start_point = get_side_center(src_rect, src_side)
        end_point = get_side_center(dst_rect, dst_side)
        draw.line([start_point, end_point], fill=line_color, width=line_wide)
        arrow_size = line_wide * arrow_size
        angle = math.atan2(end_point[1] - start_point[1], end_point[0] - start_point[0])
        p1 = (end_point[0] - arrow_size * math.cos(angle - math.pi / 6),
              end_point[1] - arrow_size * math.sin(angle - math.pi / 6))
        p2 = (end_point[0] - arrow_size * math.cos(angle + math.pi / 6),
              end_point[1] - arrow_size * math.sin(angle + math.pi / 6))
        draw.polygon([end_point, p1, p2], fill=line_color)
    return img


def highlight_region(image_path, region_to_zoom, line_color="red", line_wide=2):
    x, y, w, h = region_to_zoom
    img = Image.open(image_path).convert("RGB")
    img_copy = img.copy()

    draw = ImageDraw.Draw(img)
    draw.rectangle(
        [(x, y), (x + w, y + h)],
        outline=line_color,
        width=line_wide
    )
    return img


if __name__ == "__main__":
    import pyzjr

    x, y, w, h = select_roi_region(
        r'E:\PythonProject\CDLab\data\CLCD\test\image1\00480.png'
    )
    print(f"Final selection - {x},{y},{w},{h}")
    region_to_zoom = (x, y, w, h)
    dir_path = r'E:\PythonProject\CDLab\data\CLCD_s\test\image1'
    save_dir = './draw'
    os.makedirs(save_dir, exist_ok=True)
    image_path = pyzjr.get_image_path(dir_path, return_abs_path=True)
    for i, im_path in enumerate(image_path):
        # im = highlight_region(im_path, region_to_zoom, 'blue')
        im = plot_highlight_region(im_path, region_to_zoom, show_arrow=False, zoom_factor=1.8)
        name = os.path.basename(im_path)
        print(i, name)
        im.save(os.path.join(save_dir, name))

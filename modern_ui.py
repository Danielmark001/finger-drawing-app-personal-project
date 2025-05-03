import cv2
import numpy as np
import time
import math

def create_rounded_rectangle(image, top_left, bottom_right, radius, color, thickness=-1):
    """
    Draw a rounded rectangle on the image
    
    Parameters:
    - image: Image to draw on
    - top_left: Top-left corner coordinates (x, y)
    - bottom_right: Bottom-right corner coordinates (x, y)
    - radius: Corner radius
    - color: Rectangle color
    - thickness: Border thickness (-1 for filled)
    """
    # Draw the main rectangle
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    if thickness < 0:
        # Filled rectangle - need to draw each component separately
        # Main rectangle
        cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
        
        # Four corners
        cv2.circle(image, (x1 + radius, y1 + radius), radius, color, thickness)
        cv2.circle(image, (x2 - radius, y1 + radius), radius, color, thickness)
        cv2.circle(image, (x1 + radius, y2 - radius), radius, color, thickness)
        cv2.circle(image, (x2 - radius, y2 - radius), radius, color, thickness)
    else:
        # Outlined rectangle
        # Draw the straight lines
        cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        
        # Draw the arcs for corners
        cv2.ellipse(image, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(image, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(image, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(image, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
    
    return image

def create_button(image, text, center, width, height, idle_color, hover_color, 
                 text_color, is_hover=False, radius=10, font=cv2.FONT_HERSHEY_SIMPLEX, 
                 font_scale=0.7, font_thickness=1, icon=None):
    """
    Create a modern button with hover effect
    
    Parameters:
    - image: Image to draw the button on
    - text: Button text
    - center: Button center (x, y)
    - width, height: Button dimensions
    - idle_color: Normal state color
    - hover_color: Hover state color
    - text_color: Text color
    - is_hover: Whether the button is in hover state
    - radius: Corner radius
    - font, font_scale, font_thickness: Text parameters
    - icon: Optional tuple (icon_image, icon_width, icon_height) for button icon
    
    Returns: (image, button_bounds)
    """
    cx, cy = center
    x1 = cx - width // 2
    y1 = cy - height // 2
    x2 = cx + width // 2
    y2 = cy + height // 2
    
    # Select color based on hover state
    color = hover_color if is_hover else idle_color
    
    # Create shadow effect
    shadow_offset = 3
    shadow_color = (max(0, color[0] - 40), max(0, color[1] - 40), max(0, color[2] - 40))
    create_rounded_rectangle(image, (x1 + shadow_offset, y1 + shadow_offset), 
                            (x2 + shadow_offset, y2 + shadow_offset), radius, shadow_color, -1)
    
    # Draw button background
    create_rounded_rectangle(image, (x1, y1), (x2, y2), radius, color, -1)
    
    # Create a highlight effect on top edge
    highlight_color = (min(255, color[0] + 40), min(255, color[1] + 40), min(255, color[2] + 40))
    cv2.line(image, (x1 + radius, y1 + 2), (x2 - radius, y1 + 2), highlight_color, 2)
    
    # Add icon if provided
    if icon is not None:
        icon_img, icon_width, icon_height = icon
        
        # Calculate icon position
        icon_x = cx - icon_width // 2
        
        if text:
            # If there's text, position icon to the left of text
            text_width = cv2.getTextSize(text, font, font_scale, font_thickness)[0][0]
            spacing = 5  # Space between icon and text
            total_width = icon_width + spacing + text_width
            
            icon_x = cx - total_width // 2
            text_x = icon_x + icon_width + spacing
            
            # Position text separately
            cv2.putText(image, text, (text_x, cy + 5), font, font_scale, text_color, font_thickness)
        else:
            # Center icon if no text
            icon_x = cx - icon_width // 2
        
        icon_y = cy - icon_height // 2
        
        # Make sure icon dimensions are within bounds
        if icon_img.shape[0] != icon_height or icon_img.shape[1] != icon_width:
            icon_img = cv2.resize(icon_img, (icon_width, icon_height))
        
        # Insert icon into button
        if icon_img.shape[2] == 4:  # With alpha channel
            # Extract alpha channel and create mask
            alpha = icon_img[:, :, 3] / 255.0
            alpha = np.stack([alpha, alpha, alpha], axis=2)
            
            # Get the region where the icon will be placed
            y_end = min(icon_y + icon_height, image.shape[0])
            x_end = min(icon_x + icon_width, image.shape[1])
            
            # Calculate actual dimensions to avoid out-of-bounds
            actual_height = y_end - icon_y
            actual_width = x_end - icon_x
            
            if actual_width > 0 and actual_height > 0:
                icon_region = image[icon_y:y_end, icon_x:x_end]
                icon_part = icon_img[:actual_height, :actual_width, :3]
                alpha_part = alpha[:actual_height, :actual_width]
                
                # Blend icon with background
                image[icon_y:y_end, icon_x:x_end] = (icon_region * (1 - alpha_part) + 
                                                   icon_part * alpha_part)
        else:  # Without alpha channel
            # Regular overlay
            y_end = min(icon_y + icon_height, image.shape[0])
            x_end = min(icon_x + icon_width, image.shape[1])
            
            if x_end > icon_x and y_end > icon_y:
                image[icon_y:y_end, icon_x:x_end] = icon_img[:y_end-icon_y, :x_end-icon_x]
    elif text:
        # Position text in the center if no icon
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = cx - text_size[0] // 2
        text_y = cy + text_size[1] // 2
        cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, font_thickness)
    
    # Return the image and button bounds
    return image, ((x1, y1), (x2, y2))

def create_toggle_switch(image, center, width, height, is_on, 
                        off_color=(80, 80, 80), on_color=(0, 150, 100), 
                        knob_color=(255, 255, 255), bg_radius=None):
    """
    Create a modern toggle switch
    
    Parameters:
    - image: Image to draw on
    - center: Toggle center (x, y)
    - width, height: Toggle dimensions
    - is_on: Toggle state
    - off_color, on_color: Colors for off and on states
    - knob_color: Color of the toggle knob
    
    Returns: (image, toggle_bounds)
    """
    cx, cy = center
    x1 = cx - width // 2
    y1 = cy - height // 2
    x2 = cx + width // 2
    y2 = cy + height // 2
    
    if bg_radius is None:
        bg_radius = height // 2
    
    # Draw toggle background
    color = on_color if is_on else off_color
    create_rounded_rectangle(image, (x1, y1), (x2, y2), bg_radius, color, -1)
    
    # Calculate knob position
    knob_radius = (height - 4) // 2
    knob_x = x2 - knob_radius - 2 if is_on else x1 + knob_radius + 2
    
    # Draw knob shadow
    shadow_offset = 1
    shadow_color = (max(0, knob_color[0] - 40), max(0, knob_color[1] - 40), max(0, knob_color[2] - 40))
    cv2.circle(image, (knob_x + shadow_offset, cy + shadow_offset), knob_radius, shadow_color, -1)
    
    # Draw knob
    cv2.circle(image, (knob_x, cy), knob_radius, knob_color, -1)
    
    # Add highlight to knob
    highlight_radius = knob_radius - 3
    if highlight_radius > 0:
        highlight_color = (min(255, knob_color[0] + 40), min(255, knob_color[1] + 40), min(255, knob_color[2] + 40))
        cv2.circle(image, (knob_x - 1, cy - 1), highlight_radius, highlight_color, -1)
    
    # Return image and toggle bounds
    return image, ((x1, y1), (x2, y2))

def create_slider(image, center, width, height, value, min_value=0, max_value=100,
                 track_color=(80, 80, 80), active_color=(0, 150, 100), 
                 handle_color=(255, 255, 255), radius=5):
    """
    Create a modern slider control
    
    Parameters:
    - image: Image to draw on
    - center: Slider center (x, y)
    - width, height: Slider dimensions
    - value: Current slider value (between min_value and max_value)
    - min_value, max_value: Value range
    - track_color: Inactive track color
    - active_color: Active track color
    - handle_color: Handle color
    - radius: Corner radius
    
    Returns: (image, slider_bounds, handle_pos)
    """
    cx, cy = center
    x1 = cx - width // 2
    y1 = cy - height // 2
    x2 = cx + width // 2
    y2 = cy + height // 2
    
    # Ensure value is in range
    value = max(min_value, min(value, max_value))
    
    # Calculate handle position
    handle_pos = int(x1 + (x2 - x1) * (value - min_value) / (max_value - min_value))
    
    # Draw track background
    create_rounded_rectangle(image, (x1, y1), (x2, y2), radius, track_color, -1)
    
    # Draw active part of track
    if handle_pos > x1:
        create_rounded_rectangle(image, (x1, y1), (handle_pos, y2), radius, active_color, -1)
    
    # Draw handle
    handle_radius = height
    cv2.circle(image, (handle_pos, cy), handle_radius, handle_color, -1)
    
    # Add highlight to handle
    highlight_radius = handle_radius - 3
    if highlight_radius > 0:
        highlight_color = (min(255, handle_color[0] + 40), min(255, handle_color[1] + 40), min(255, handle_color[2] + 40))
        cv2.circle(image, (handle_pos - 1, cy - 1), highlight_radius, highlight_color, -1)
    
    # Return image, slider bounds, and handle position
    return image, ((x1, y1), (x2, y2)), handle_pos

def create_color_palette(width, height, palette_type="spectrum"):
    """
    Create a modern color palette image
    
    Parameters:
    - width, height: Palette dimensions
    - palette_type: Type of palette ("spectrum", "rgb", "hsv")
    
    Returns: Image containing color palette
    """
    palette = np.zeros((height, width, 3), dtype=np.uint8)
    
    if palette_type == "spectrum":
        # Create rainbow spectrum
        for x in range(width):
            # Map x to hue (0-179 for OpenCV)
            hue = int(x * 180 / width)
            # Create a column with full saturation and value
            color = np.full((height, 1, 3), (hue, 255, 255), dtype=np.uint8)
            # Convert to BGR
            color_bgr = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)
            # Add to palette
            palette[:, x:x+1] = color_bgr
    
    elif palette_type == "hsv":
        # Create HSV palette - hue horizontally, saturation vertically
        for y in range(height):
            for x in range(width):
                hue = int(x * 180 / width)
                sat = int((height - y) * 255 / height)
                val = 255
                palette[y, x] = cv2.cvtColor(np.uint8([[[hue, sat, val]]]), cv2.COLOR_HSV2BGR)[0, 0]
    
    elif palette_type == "rgb":
        # Create gradient from primary colors
        num_segments = 6
        segment_width = width // num_segments
        
        colors = [
            (0, 0, 255),      # Red
            (0, 255, 255),    # Yellow
            (0, 255, 0),      # Green
            (255, 255, 0),    # Cyan
            (255, 0, 0),      # Blue
            (255, 0, 255)     # Magenta
        ]
        
        for i in range(num_segments):
            start_x = i * segment_width
            end_x = (i + 1) * segment_width if i < num_segments - 1 else width
            
            start_color = colors[i]
            end_color = colors[(i + 1) % num_segments]
            
            for x in range(start_x, end_x):
                # Calculate gradient factor
                t = (x - start_x) / (end_x - start_x)
                
                # Interpolate between colors
                r = int(start_color[0] * (1 - t) + end_color[0] * t)
                g = int(start_color[1] * (1 - t) + end_color[1] * t)
                b = int(start_color[2] * (1 - t) + end_color[2] * t)
                
                # Fill the column
                palette[:, x] = (b, g, r)
    
    else:  # Default grayscale
        for x in range(width):
            gray_value = int(x * 255 / width)
            palette[:, x] = (gray_value, gray_value, gray_value)
    
    return palette

def create_advanced_color_picker(width, height):
    """
    Create an advanced color picker with hue selector and saturation/value panel
    
    Parameters:
    - width, height: Overall dimensions
    
    Returns: (image, hue_picker_area, sv_picker_area)
    """
    # Create main image
    picker = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Define areas for hue and SV pickers
    hue_height = height // 6
    sv_height = height - hue_height - 10  # Spacing between pickers
    
    hue_area = (0, 0, width, hue_height)
    sv_area = (0, hue_height + 10, width, height)
    
    # Create hue picker (rainbow)
    for x in range(width):
        hue = int(x * 180 / width)
        hue_color = np.full((hue_height, 1, 3), (hue, 255, 255), dtype=np.uint8)
        hue_color_bgr = cv2.cvtColor(hue_color, cv2.COLOR_HSV2BGR)
        picker[:hue_height, x:x+1] = hue_color_bgr
    
    # Add border to hue picker
    cv2.rectangle(picker, (0, 0), (width-1, hue_height-1), (150, 150, 150), 1)
    
    # Create saturation/value picker (2D gradient)
    selected_hue = 0  # Default hue is red
    for y in range(sv_height):
        for x in range(width):
            # Map x to saturation (0-255)
            s = int(x * 255 / width)
            # Map y inversely to value (255-0)
            v = int((1.0 - y / sv_height) * 255)
            
            # Create HSV color and convert to BGR
            color_hsv = np.uint8([[[selected_hue, s, v]]])
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)
            
            # Set pixel color
            picker[hue_height+10+y, x] = color_bgr[0, 0]
    
    # Add border to SV picker
    cv2.rectangle(picker, (0, hue_height+10), (width-1, height-1), (150, 150, 150), 1)
    
    return picker, hue_area, sv_area

def create_color_swatch(image, center, color, size=30, selected=False, border_color=(200, 200, 200)):
    """
    Create a color swatch for the palette
    
    Parameters:
    - image: Image to draw on
    - center: Swatch center (x, y)
    - color: The color to display
    - size: Size of the swatch
    - selected: Whether the swatch is selected
    - border_color: Color of the border
    
    Returns: (image, swatch_bounds)
    """
    cx, cy = center
    half_size = size // 2
    x1, y1 = cx - half_size, cy - half_size
    x2, y2 = cx + half_size, cy + half_size
    
    # Draw shadow
    shadow_offset = 2
    shadow_color = (30, 30, 30)
    shadow_rect = ((x1 + shadow_offset, y1 + shadow_offset), (x2 + shadow_offset, y2 + shadow_offset))
    cv2.rectangle(image, shadow_rect[0], shadow_rect[1], shadow_color, -1)
    
    # Draw color swatch
    cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
    
    # Draw border
    border_thickness = 2 if selected else 1
    cv2.rectangle(image, (x1, y1), (x2, y2), border_color, border_thickness)
    
    # If selected, add extra highlight
    if selected:
        highlight_padding = 3
        highlight_rect = ((x1 - highlight_padding, y1 - highlight_padding), 
                          (x2 + highlight_padding, y2 + highlight_padding))
        cv2.rectangle(image, highlight_rect[0], highlight_rect[1], (255, 255, 255), 1)
    
    return image, ((x1, y1), (x2, y2))

def create_toolbar(image, x, y, width, height, tools, active_tool, 
                  bg_color=(40, 40, 50), border_color=(70, 70, 80)):
    """
    Create a modern floating toolbar
    
    Parameters:
    - image: Image to draw on
    - x, y: Top-left corner of toolbar
    - width, height: Toolbar dimensions
    - tools: List of tool dictionaries with keys: name, icon
    - active_tool: Name of the active tool
    - bg_color, border_color: Toolbar colors
    
    Returns: (image, toolbar_area, tool_regions)
    """
    # Create toolbar background
    create_rounded_rectangle(image, (x, y), (x + width, y + height), 10, bg_color, -1)
    create_rounded_rectangle(image, (x, y), (x + width, y + height), 10, border_color, 2)
    
    # Add subtle gradient
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        # Gradient from darkest at bottom to lighter at top
        alpha = 1.0 - (i / height) * 0.7
        gradient[i, :] = np.array(bg_color) * alpha
    
    # Create mask for the rounded corners
    mask = np.zeros((height, width), dtype=np.uint8)
    create_rounded_rectangle(mask, (0, 0), (width, height), 10, (255, 255, 255), -1)
    
    # Blend with original image
    roi = image[y:y+height, x:x+width]
    for c in range(3):
        roi[:, :, c] = np.where(mask > 0, 
                               (roi[:, :, c] * 0.7 + gradient[:, :, c] * 0.3), 
                               roi[:, :, c])
    
    # Draw tool buttons
    num_tools = len(tools)
    tool_regions = []
    
    if num_tools > 0:
        tool_width = (width - 20) // num_tools  # Allow some padding
        tool_height = height - 20  # Allow some padding
        
        for i, tool in enumerate(tools):
            tool_x = x + 10 + i * tool_width
            tool_y = y + 10
            
            is_active = tool["name"] == active_tool
            tool_color = (60, 60, 100) if is_active else (50, 50, 70)
            border_color = (100, 100, 255) if is_active else (70, 70, 90)
            
            # Tool button
            create_rounded_rectangle(image, (tool_x, tool_y), 
                                    (tool_x + tool_width - 5, tool_y + tool_height), 
                                    5, tool_color, -1)
            
            if is_active:
                # Add highlight for active tool
                create_rounded_rectangle(image, (tool_x, tool_y), 
                                        (tool_x + tool_width - 5, tool_y + tool_height), 
                                        5, border_color, 2)
            
            # Add tool icon or text
            tool_center_x = tool_x + (tool_width - 5) // 2
            tool_center_y = tool_y + tool_height // 2
            
            if "icon" in tool and tool["icon"] is not None:
                # Add icon
                icon_img = tool["icon"]
                icon_size = min(tool_width - 10, tool_height - 10)
                
                # Resize icon if needed
                if icon_img.shape[0] != icon_size or icon_img.shape[1] != icon_size:
                    icon_img = cv2.resize(icon_img, (icon_size, icon_size))
                
                # Calculate icon position
                icon_x = tool_center_x - icon_size // 2
                icon_y = tool_center_y - icon_size // 2
                
                # Draw icon
                if icon_img.shape[2] == 4:  # With alpha
                    # Extract alpha channel
                    alpha = icon_img[:, :, 3] / 255.0
                    alpha = np.stack([alpha, alpha, alpha], axis=2)
                    
                    # Blend icon with background
                    icon_region = image[icon_y:icon_y+icon_size, icon_x:icon_x+icon_size]
                    image[icon_y:icon_y+icon_size, icon_x:icon_x+icon_size] = (
                        icon_region * (1 - alpha) + icon_img[:, :, :3] * alpha
                    )
                else:
                    # Just copy the icon
                    image[icon_y:icon_y+icon_size, icon_x:icon_x+icon_size] = icon_img
            else:
                # Add text
                text = tool["name"].capitalize()
                text_color = (255, 255, 255) if is_active else (200, 200, 200)
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                
                text_x = tool_center_x - text_size[0] // 2
                text_y = tool_center_y + text_size[1] // 2
                
                cv2.putText(image, text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            
            # Store tool region
            tool_regions.append({
                "name": tool["name"],
                "region": ((tool_x, tool_y), (tool_x + tool_width - 5, tool_y + tool_height))
            })
    
    return image, ((x, y), (x + width, y + height)), tool_regions

def create_tooltip(image, text, position, bg_color=(40, 40, 50), text_color=(255, 255, 255),
                  font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6, padding=5):
    """
    Draw a modern tooltip with arrow
    
    Parameters:
    - image: Image to draw on
    - text: Tooltip text
    - position: Position to show tooltip (x, y)
    - bg_color, text_color: Tooltip colors
    
    Returns: image with tooltip
    """
    x, y = position
    
    # Measure text
    text_size, baseline = cv2.getTextSize(text, font, font_scale, 1)
    text_width, text_height = text_size
    
    # Calculate tooltip dimensions
    tooltip_width = text_width + padding * 2
    tooltip_height = text_height + padding * 2
    
    # Position tooltip above the point
    tooltip_x = max(0, x - tooltip_width // 2)
    tooltip_y = max(0, y - tooltip_height - 15)  # Leave space for arrow
    
    # Ensure tooltip is within image bounds
    if tooltip_x + tooltip_width > image.shape[1]:
        tooltip_x = image.shape[1] - tooltip_width
    
    # Draw tooltip background
    create_rounded_rectangle(image, (tooltip_x, tooltip_y), 
                            (tooltip_x + tooltip_width, tooltip_y + tooltip_height), 
                            5, bg_color, -1)
    
    # Draw arrow pointing to position
    arrow_x = x
    arrow_y = tooltip_y + tooltip_height
    
    # Ensure arrow is within tooltip bounds
    arrow_x = max(tooltip_x + 10, min(tooltip_x + tooltip_width - 10, arrow_x))
    
    # Arrow points
    arrow_points = np.array([
        [arrow_x, arrow_y + 10],  # Bottom point
        [arrow_x - 8, arrow_y],   # Left point
        [arrow_x + 8, arrow_y]    # Right point
    ], np.int32)
    
    # Draw filled arrow
    cv2.fillPoly(image, [arrow_points], bg_color)
    
    # Draw text
    text_x = tooltip_x + padding
    text_y = tooltip_y + padding + text_height
    
    cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, 1)
    
    return image

def create_layers_panel(image, x, y, width, height, layers, active_layer,
                       bg_color=(40, 40, 50), border_color=(70, 70, 80)):
    """
    Create a modern layers panel
    
    Parameters:
    - image: Image to draw on
    - x, y: Top-left corner of panel
    - width, height: Panel dimensions
    - layers: List of layer dictionaries with keys: name, visible, thumbnail
    - active_layer: Index of the active layer
    
    Returns: (image, panel_area, layer_regions, button_regions)
    """
    # Draw panel background
    create_rounded_rectangle(image, (x, y), (x + width, y + height), 10, bg_color, -1)
    create_rounded_rectangle(image, (x, y), (x + width, y + height), 10, border_color, 2)
    
    # Add title
    title = "Layers"
    cv2.putText(image, title, (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 1)
    
    # Draw separator
    cv2.line(image, (x + 10, y + 35), (x + width - 10, y + 35), (100, 100, 120), 1)
    
    # Add layer controls
    button_size = 25
    button_margin = 5
    button_y = y + 45
    button_regions = []
    
    # Add new layer button
    new_button_x = x + 10
    button_color = (60, 100, 60)
    create_rounded_rectangle(image, (new_button_x, button_y), 
                            (new_button_x + button_size, button_y + button_size), 
                            5, button_color, -1)
    cv2.putText(image, "+", (new_button_x + 8, button_y + 18), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1)
    button_regions.append({
        "action": "new_layer",
        "region": ((new_button_x, button_y), (new_button_x + button_size, button_y + button_size))
    })
    
    # Delete layer button
    delete_button_x = new_button_x + button_size + button_margin
    button_color = (100, 60, 60)
    create_rounded_rectangle(image, (delete_button_x, button_y), 
                            (delete_button_x + button_size, button_y + button_size), 
                            5, button_color, -1)
    cv2.putText(image, "-", (delete_button_x + 10, button_y + 18), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1)
    button_regions.append({
        "action": "delete_layer",
        "region": ((delete_button_x, button_y), (delete_button_x + button_size, button_y + button_size))
    })
    
    # Move layer up button
    up_button_x = delete_button_x + button_size + button_margin
    button_color = (60, 60, 100)
    create_rounded_rectangle(image, (up_button_x, button_y), 
                            (up_button_x + button_size, button_y + button_size), 
                            5, button_color, -1)
    cv2.putText(image, "↑", (up_button_x + 8, button_y + 18), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1)
    button_regions.append({
        "action": "move_up",
        "region": ((up_button_x, button_y), (up_button_x + button_size, button_y + button_size))
    })
    
    # Move layer down button
    down_button_x = up_button_x + button_size + button_margin
    button_color = (60, 60, 100)
    create_rounded_rectangle(image, (down_button_x, button_y), 
                            (down_button_x + button_size, button_y + button_size), 
                            5, button_color, -1)
    cv2.putText(image, "↓", (down_button_x + 8, button_y + 18), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1)
    button_regions.append({
        "action": "move_down",
        "region": ((down_button_x, button_y), (down_button_x + button_size, button_y + button_size))
    })
    
    # Draw layer list
    layer_height = 40
    layer_margin = 5
    layer_regions = []
    
    start_y = button_y + button_size + 15
    max_visible_layers = (height - (start_y - y) - 10) // (layer_height + layer_margin)
    
    # Show layers in reverse order (top layer first)
    visible_layers = layers[::-1][:max_visible_layers]
    
    for i, layer in enumerate(visible_layers):
        layer_index = len(layers) - 1 - i
        is_active = layer_index == active_layer
        
        layer_y = start_y + i * (layer_height + layer_margin)
        
        # Draw layer background
        bg_color = (70, 70, 100) if is_active else (50, 50, 70)
        create_rounded_rectangle(image, (x + 10, layer_y), 
                                (x + width - 10, layer_y + layer_height), 
                                5, bg_color, -1)
        
        # Draw active layer highlight
        if is_active:
            highlight_color = (100, 100, 200)
            create_rounded_rectangle(image, (x + 10, layer_y), 
                                    (x + width - 10, layer_y + layer_height), 
                                    5, highlight_color, 2)
        
        # Draw layer name
        layer_name = layer.get("name", f"Layer {layer_index + 1}")
        cv2.putText(image, layer_name, (x + 15, layer_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
        
        # Draw visibility icon
        eye_x = x + width - 30
        eye_y = layer_y + layer_height // 2
        
        if layer.get("visible", True):
            # Eye open icon
            cv2.circle(image, (eye_x, eye_y), 7, (200, 200, 200), 1)
            cv2.circle(image, (eye_x, eye_y), 3, (200, 200, 200), -1)
        else:
            # Eye closed icon
            cv2.line(image, (eye_x - 7, eye_y), (eye_x + 7, eye_y), (150, 150, 150), 1)
        
        # Add layer thumbnail if available
        thumbnail = layer.get("thumbnail", None)
        if thumbnail is not None:
            thumb_size = min(30, layer_height - 10)
            thumb_x = x + 60
            thumb_y = layer_y + (layer_height - thumb_size) // 2
            
            # Resize thumbnail if needed
            if thumbnail.shape[0] != thumb_size or thumbnail.shape[1] != thumb_size:
                thumbnail = cv2.resize(thumbnail, (thumb_size, thumb_size))
            
            # Draw thumbnail with border
            cv2.rectangle(image, (thumb_x - 1, thumb_y - 1), 
                         (thumb_x + thumb_size + 1, thumb_y + thumb_size + 1), 
                         (70, 70, 70), 1)
            
            # Place thumbnail
            if thumbnail.shape[2] == 4:  # With alpha
                # Handle alpha channel
                alpha = thumbnail[:, :, 3] / 255.0
                alpha = np.stack([alpha, alpha, alpha], axis=2)
                
                roi = image[thumb_y:thumb_y+thumb_size, thumb_x:thumb_x+thumb_size]
                image[thumb_y:thumb_y+thumb_size, thumb_x:thumb_x+thumb_size] = (
                    roi * (1 - alpha) + thumbnail[:, :, :3] * alpha
                )
            else:
                image[thumb_y:thumb_y+thumb_size, thumb_x:thumb_x+thumb_size] = thumbnail
        
        # Store layer region
        layer_regions.append({
            "layer_index": layer_index,
            "region": ((x + 10, layer_y), (x + width - 35, layer_y + layer_height)),
            "visibility_region": ((eye_x - 8, eye_y - 8), (eye_x + 8, eye_y + 8))
        })
    
    return image, ((x, y), (x + width, y + height)), layer_regions, button_regions

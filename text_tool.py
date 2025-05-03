import cv2
import numpy as np
import math

class TextTool:
    """Text tool for adding text to drawings."""
    
    def __init__(self):
        self.text = ""
        self.position = (100, 100)
        self.color = (255, 255, 255)  # Default white in BGR
        self.font_size = 1.0
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.thickness = 2
        self.is_editing = False
        self.is_dragging = False
        self.drag_offset = (0, 0)
        self.rotation = 0  # Rotation in degrees
        self.available_fonts = [
            {"name": "Default", "id": cv2.FONT_HERSHEY_SIMPLEX},
            {"name": "Plain", "id": cv2.FONT_HERSHEY_PLAIN},
            {"name": "Duplex", "id": cv2.FONT_HERSHEY_DUPLEX},
            {"name": "Complex", "id": cv2.FONT_HERSHEY_COMPLEX},
            {"name": "Triplex", "id": cv2.FONT_HERSHEY_TRIPLEX},
            {"name": "Complex Small", "id": cv2.FONT_HERSHEY_COMPLEX_SMALL},
            {"name": "Script", "id": cv2.FONT_HERSHEY_SCRIPT_SIMPLEX},
            {"name": "Script Complex", "id": cv2.FONT_HERSHEY_SCRIPT_COMPLEX}
        ]
        self.current_font_index = 0
    
    def set_text(self, text):
        """Set the text content."""
        self.text = text
    
    def set_position(self, position):
        """Set the text position."""
        self.position = position
    
    def set_color(self, color):
        """Set the text color."""
        self.color = color
    
    def set_font_size(self, size):
        """Set the font size."""
        self.font_size = max(0.5, min(5.0, size))
    
    def set_font(self, font_index):
        """Set the font by index."""
        if 0 <= font_index < len(self.available_fonts):
            self.current_font_index = font_index
            self.font = self.available_fonts[font_index]["id"]
    
    def set_thickness(self, thickness):
        """Set the text thickness."""
        self.thickness = max(1, min(5, thickness))
    
    def set_rotation(self, rotation):
        """Set the text rotation in degrees."""
        self.rotation = rotation % 360
    
    def start_editing(self):
        """Start editing the text."""
        self.is_editing = True
    
    def stop_editing(self):
        """Stop editing the text."""
        self.is_editing = False
    
    def start_dragging(self, cursor_pos):
        """Start dragging the text."""
        self.is_dragging = True
        x, y = self.position
        cx, cy = cursor_pos
        self.drag_offset = (x - cx, y - cy)
    
    def update_drag(self, cursor_pos):
        """Update the position while dragging."""
        if self.is_dragging:
            cx, cy = cursor_pos
            dx, dy = self.drag_offset
            self.position = (cx + dx, cy + dy)
    
    def stop_dragging(self):
        """Stop dragging the text."""
        self.is_dragging = False
    
    def get_text_area(self):
        """Get the area occupied by the text."""
        if not self.text:
            return ((0, 0), (0, 0))
        
        # Calculate text size
        text_size, _ = cv2.getTextSize(self.text, self.font, self.font_size, self.thickness)
        width, height = text_size
        
        # Calculate bounding box
        if self.rotation == 0:
            x, y = self.position
            return ((x, y - height), (x + width, y))
        else:
            # For rotated text, we need a larger bounding box
            x, y = self.position
            # Use the diagonal of the rectangle as the radius
            radius = math.sqrt(width**2 + height**2) / 2
            
            # Calculate the center of the text
            center_x = x + width / 2
            center_y = y - height / 2
            
            # Calculate the corners
            top_left = (center_x - radius, center_y - radius)
            bottom_right = (center_x + radius, center_y + radius)
            
            return (top_left, bottom_right)
    
    def contains_point(self, point):
        """Check if a point is within the text area."""
        (x1, y1), (x2, y2) = self.get_text_area()
        px, py = point
        
        # Add some padding for easier selection
        padding = 10
        return (x1 - padding <= px <= x2 + padding and 
                y1 - padding <= py <= y2 + padding)
    
    def add_to_layer(self, layer):
        """Add the text to a layer."""
        if not self.text:
            return layer
        
        # Create a temporary overlay for the text
        h, w = layer.shape[:2]
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Get current font
        font = self.font
        
        if self.rotation == 0:
            # Add text directly for non-rotated text
            cv2.putText(overlay, self.text, self.position, font, 
                       self.font_size, (*self.color, 255), self.thickness)
        else:
            # For rotated text, we need to use a rotation matrix
            text_size, _ = cv2.getTextSize(self.text, font, self.font_size, self.thickness)
            text_width, text_height = text_size
            
            # Create a temporary image just for the text
            text_img = np.zeros((text_height * 2, text_width * 2, 4), dtype=np.uint8)
            
            # Position text in the center of the temporary image
            text_pos = (text_width // 2, text_height + text_height // 2)
            cv2.putText(text_img, self.text, text_pos, font, 
                       self.font_size, (*self.color, 255), self.thickness)
            
            # Rotate the text image
            center = (text_width, text_height)
            rot_mat = cv2.getRotationMatrix2D(center, self.rotation, 1.0)
            rotated_text = cv2.warpAffine(text_img, rot_mat, (text_width * 2, text_height * 2), 
                                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
            
            # Calculate position to place the rotated text
            x, y = self.position
            place_x = max(0, min(w - rotated_text.shape[1], x - text_width))
            place_y = max(0, min(h - rotated_text.shape[0], y - text_height))
            
            # Place the rotated text on the overlay
            overlay_roi = overlay[place_y:place_y+rotated_text.shape[0], 
                                 place_x:place_x+rotated_text.shape[1]]
            
            # Make sure ROI is not out of bounds
            if overlay_roi.shape[:2] == rotated_text.shape[:2]:
                # Extract alpha from rotated text
                alpha = rotated_text[:, :, 3] / 255.0
                alpha = np.stack([alpha, alpha, alpha], axis=2)
                
                # Update RGB channels
                overlay_roi[:, :, :3] = rotated_text[:, :, :3] * alpha
                
                # Update alpha channel
                overlay_roi[:, :, 3] = rotated_text[:, :, 3]
        
        # Blend the overlay with the layer
        alpha = overlay[:, :, 3] / 255.0
        alpha = np.stack([alpha, alpha, alpha], axis=2)
        
        # Update RGB
        layer[:, :, :3] = layer[:, :, :3] * (1 - alpha) + overlay[:, :, :3] * alpha
        
        # Update alpha channel (if layer has one)
        if layer.shape[2] == 4:
            layer_alpha = layer[:, :, 3] / 255.0
            new_alpha = layer_alpha + (overlay[:, :, 3] / 255.0) * (1 - layer_alpha)
            layer[:, :, 3] = (new_alpha * 255).astype(np.uint8)
        
        return layer
    
    def render_editor(self, image, panel_x, panel_y, panel_width, panel_height):
        """Render the text editor panel."""
        # Draw panel background
        cv2.rectangle(image, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (40, 40, 50), -1)
        cv2.rectangle(image, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (70, 70, 90), 2)
        
        # Add title
        cv2.putText(image, "Text Tool", (panel_x + 10, panel_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Text input area
        input_y = panel_y + 50
        cv2.rectangle(image, (panel_x + 10, input_y), 
                     (panel_x + panel_width - 10, input_y + 40), 
                     (30, 30, 40), -1)
        cv2.rectangle(image, (panel_x + 10, input_y), 
                     (panel_x + panel_width - 10, input_y + 40), 
                     (100, 100, 120), 1)
        
        # Show current text
        display_text = self.text
        if not display_text:
            display_text = "Enter text here..."
        
        # Truncate if too long
        if len(display_text) > 30:
            display_text = display_text[:27] + "..."
        
        text_color = (200, 200, 200) if self.text else (120, 120, 120)
        cv2.putText(image, display_text, (panel_x + 15, input_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        
        # Add cursor if editing
        if self.is_editing:
            text_width = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
            cursor_x = panel_x + 15 + text_width
            cv2.line(image, (cursor_x, input_y + 10), (cursor_x, input_y + 30), 
                    (200, 200, 200), 1)
        
        # Add input hint
        hint_y = input_y + 50
        cv2.putText(image, "Click to edit text", (panel_x + 10, hint_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Font selector
        font_y = hint_y + 30
        cv2.putText(image, "Font:", (panel_x + 10, font_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Font dropdown
        dropdown_x = panel_x + 70
        dropdown_width = 180
        dropdown_height = 30
        
        cv2.rectangle(image, (dropdown_x, font_y - 20), 
                     (dropdown_x + dropdown_width, font_y + 10), 
                     (60, 60, 80), -1)
        cv2.rectangle(image, (dropdown_x, font_y - 20), 
                     (dropdown_x + dropdown_width, font_y + 10), 
                     (100, 100, 120), 1)
        
        # Show current font
        font_name = self.available_fonts[self.current_font_index]["name"]
        cv2.putText(image, font_name, (dropdown_x + 10, font_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        
        # Draw dropdown arrow
        arrow_x = dropdown_x + dropdown_width - 20
        arrow_y = font_y - 5
        cv2.line(image, (arrow_x, arrow_y), (arrow_x + 10, arrow_y), (200, 200, 200), 1)
        cv2.line(image, (arrow_x, arrow_y), (arrow_x + 5, arrow_y + 5), (200, 200, 200), 1)
        cv2.line(image, (arrow_x + 10, arrow_y), (arrow_x + 5, arrow_y + 5), (200, 200, 200), 1)
        
        # Font size controls
        size_y = font_y + 40
        cv2.putText(image, "Size:", (panel_x + 10, size_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Size slider
        slider_x = panel_x + 70
        slider_width = 150
        
        cv2.rectangle(image, (slider_x, size_y - 10), 
                     (slider_x + slider_width, size_y), 
                     (70, 70, 70), -1)
        
        # Calculate handle position (size range 0.5 - 5.0)
        size_range = 4.5  # 5.0 - 0.5
        handle_pos = int(slider_x + (self.font_size - 0.5) / size_range * slider_width)
        
        # Draw filled part
        cv2.rectangle(image, (slider_x, size_y - 10), 
                     (handle_pos, size_y), 
                     (100, 100, 180), -1)
        
        # Draw handle
        cv2.circle(image, (handle_pos, size_y - 5), 6, (200, 200, 200), -1)
        cv2.circle(image, (handle_pos, size_y - 5), 6, (70, 70, 70), 1)
        
        # Show size value
        size_text = f"{self.font_size:.1f}"
        cv2.putText(image, size_text, (slider_x + slider_width + 10, size_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Text color
        color_y = size_y + 40
        cv2.putText(image, "Color:", (panel_x + 10, color_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Color swatch
        swatch_x = panel_x + 70
        swatch_size = 25
        
        cv2.rectangle(image, (swatch_x, color_y - 20), 
                     (swatch_x + swatch_size, color_y + 5), 
                     self.color, -1)
        cv2.rectangle(image, (swatch_x, color_y - 20), 
                     (swatch_x + swatch_size, color_y + 5), 
                     (200, 200, 200), 1)
        
        # RGB controls
        rgb_x = swatch_x + swatch_size + 15
        
        # R control
        cv2.rectangle(image, (rgb_x, color_y - 20), 
                     (rgb_x + 25, color_y + 5), 
                     (0, 0, 255), -1)  # Red in BGR
        cv2.putText(image, "R", (rgb_x + 8, color_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # G control
        g_x = rgb_x + 30
        cv2.rectangle(image, (g_x, color_y - 20), 
                     (g_x + 25, color_y + 5), 
                     (0, 255, 0), -1)  # Green in BGR
        cv2.putText(image, "G", (g_x + 8, color_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # B control
        b_x = g_x + 30
        cv2.rectangle(image, (b_x, color_y - 20), 
                     (b_x + 25, color_y + 5), 
                     (255, 0, 0), -1)  # Blue in BGR
        cv2.putText(image, "B", (b_x + 8, color_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # White and black controls
        w_x = b_x + 35
        cv2.rectangle(image, (w_x, color_y - 20), 
                     (w_x + 25, color_y + 5), 
                     (255, 255, 255), -1)  # White
        cv2.rectangle(image, (w_x, color_y - 20), 
                     (w_x + 25, color_y + 5), 
                     (100, 100, 100), 1)
        
        b_x = w_x + 30
        cv2.rectangle(image, (b_x, color_y - 20), 
                     (b_x + 25, color_y + 5), 
                     (0, 0, 0), -1)  # Black
        cv2.rectangle(image, (b_x, color_y - 20), 
                     (b_x + 25, color_y + 5), 
                     (100, 100, 100), 1)
        
        # Rotation control
        rotation_y = color_y + 40
        cv2.putText(image, "Rotation:", (panel_x + 10, rotation_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Rotation slider
        rot_slider_x = panel_x + 90
        rot_slider_width = 150
        
        cv2.rectangle(image, (rot_slider_x, rotation_y - 10), 
                     (rot_slider_x + rot_slider_width, rotation_y), 
                     (70, 70, 70), -1)
        
        # Calculate handle position (rotation range 0-360)
        rot_handle_pos = int(rot_slider_x + (self.rotation / 360.0) * rot_slider_width)
        
        # Draw filled part
        cv2.rectangle(image, (rot_slider_x, rotation_y - 10), 
                     (rot_handle_pos, rotation_y), 
                     (100, 100, 180), -1)
        
        # Draw handle
        cv2.circle(image, (rot_handle_pos, rotation_y - 5), 6, (200, 200, 200), -1)
        cv2.circle(image, (rot_handle_pos, rotation_y - 5), 6, (70, 70, 70), 1)
        
        # Show rotation value
        rot_text = f"{self.rotation}°"
        cv2.putText(image, rot_text, (rot_slider_x + rot_slider_width + 10, rotation_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Preview area
        preview_y = rotation_y + 40
        preview_height = panel_height - (preview_y - panel_y) - 10
        
        cv2.rectangle(image, (panel_x + 10, preview_y), 
                     (panel_x + panel_width - 10, preview_y + preview_height), 
                     (30, 30, 40), -1)
        cv2.rectangle(image, (panel_x + 10, preview_y), 
                     (panel_x + panel_width - 10, preview_y + preview_height), 
                     (70, 70, 90), 1)
        
        # Add preview label
        cv2.putText(image, "Preview:", (panel_x + 15, preview_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Draw text preview
        if self.text:
            # Calculate text size
            text_size, _ = cv2.getTextSize(self.text, self.font, self.font_size, self.thickness)
            text_width, text_height = text_size
            
            # Center text in preview
            preview_center_x = panel_x + panel_width // 2
            preview_center_y = preview_y + preview_height // 2
            
            if self.rotation == 0:
                # Simple centered text
                text_x = preview_center_x - text_width // 2
                text_y = preview_center_y + text_height // 2
                
                cv2.putText(image, self.text, (text_x, text_y), self.font, 
                           self.font_size, self.color, self.thickness)
            else:
                # Create a temporary image for rotated text
                temp = np.zeros((preview_height, panel_width - 20, 3), dtype=np.uint8)
                temp.fill(30)  # Same as preview background
                
                # Put text in center
                temp_center_x = temp.shape[1] // 2
                temp_center_y = temp.shape[0] // 2
                
                text_x = temp_center_x - text_width // 2
                text_y = temp_center_y + text_height // 2
                
                cv2.putText(temp, self.text, (text_x, text_y), self.font, 
                           self.font_size, self.color, self.thickness)
                
                # Rotate around center
                rot_mat = cv2.getRotationMatrix2D((temp_center_x, temp_center_y), 
                                                 self.rotation, 1.0)
                rotated = cv2.warpAffine(temp, rot_mat, (temp.shape[1], temp.shape[0]))
                
                # Place in preview area
                image[preview_y:preview_y+preview_height, 
                     panel_x+10:panel_x+panel_width-10] = rotated
        else:
            # Show placeholder text
            placeholder = "Text preview will appear here"
            cv2.putText(image, placeholder, (panel_x + 50, preview_y + preview_height // 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        # Define interactive regions
        regions = {
            "text_input": ((panel_x + 10, input_y), (panel_x + panel_width - 10, input_y + 40)),
            "font_dropdown": ((dropdown_x, font_y - 20), (dropdown_x + dropdown_width, font_y + 10)),
            "size_slider": ((slider_x, size_y - 15), (slider_x + slider_width, size_y + 5)),
            "color_swatch": ((swatch_x, color_y - 20), (swatch_x + swatch_size, color_y + 5)),
            "color_r": ((rgb_x, color_y - 20), (rgb_x + 25, color_y + 5)),
            "color_g": ((g_x, color_y - 20), (g_x + 25, color_y + 5)),
            "color_b": ((b_x, color_y - 20), (b_x + 25, color_y + 5)),
            "color_white": ((w_x, color_y - 20), (w_x + 25, color_y + 5)),
            "color_black": ((b_x, color_y - 20), (b_x + 25, color_y + 5)),
            "rotation_slider": ((rot_slider_x, rotation_y - 15), (rot_slider_x + rot_slider_width, rotation_y + 5))
        }
        
        return image, regions
    
    def draw_text_preview(self, image):
        """Draw a preview of the text at its current position."""
        if not self.text:
            return image
        
        # Create a copy to avoid modifying the original
        preview = image.copy()
        
        if self.rotation == 0:
            # Draw text directly
            cv2.putText(preview, self.text, self.position, self.font, 
                       self.font_size, self.color, self.thickness)
        else:
            # Calculate text size
            text_size, _ = cv2.getTextSize(self.text, self.font, self.font_size, self.thickness)
            text_width, text_height = text_size
            
            # Create a temporary image for the text
            temp = np.zeros((text_height * 3, text_width * 3, 4), dtype=np.uint8)
            
            # Position text in center
            text_pos = (text_width, text_height * 2)
            cv2.putText(temp, self.text, text_pos, self.font, 
                       self.font_size, (*self.color, 255), self.thickness)
            
            # Rotate around center
            center = (text_width * 1.5, text_height * 1.5)
            rot_mat = cv2.getRotationMatrix2D(center, self.rotation, 1.0)
            rotated = cv2.warpAffine(temp, rot_mat, (temp.shape[1], temp.shape[0]), 
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
            
            # Calculate placement position
            x, y = self.position
            pos_x = max(0, x - text_width)
            pos_y = max(0, y - text_height)
            
            # Make sure we don't go out of bounds
            h, w = image.shape[:2]
            max_h = min(h - pos_y, rotated.shape[0])
            max_w = min(w - pos_x, rotated.shape[1])
            
            if max_h > 0 and max_w > 0:
                # Extract alpha channel
                alpha = rotated[:max_h, :max_w, 3] / 255.0
                alpha = np.stack([alpha, alpha, alpha], axis=2)
                
                # Blend with the preview image
                preview_region = preview[pos_y:pos_y+max_h, pos_x:pos_x+max_w]
                preview_region[:] = preview_region * (1 - alpha) + rotated[:max_h, :max_w, :3] * alpha
        
        # Draw a bounding box around the text
        (x1, y1), (x2, y2) = self.get_text_area()
        cv2.rectangle(preview, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 1)
        
        return preview

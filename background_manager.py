import cv2
import numpy as np
import os
import glob
import math

class BackgroundManager:
    """Manages backgrounds for the drawing app."""
    
    def __init__(self, bg_dir="backgrounds"):
        self.bg_dir = bg_dir
        self.backgrounds = []
        self.categories = set()
        self.active_category = "All"
        self.current_page = 0
        self.items_per_page = 6
        self.current_background = None
        self.background_opacity = 0.5  # Default opacity
        
        # Create backgrounds directory if it doesn't exist
        if not os.path.exists(bg_dir):
            os.makedirs(bg_dir)
            print(f"Created backgrounds directory: {bg_dir}")
            
            # Create some example categories
            categories = ["patterns", "textures", "solid", "gradient"]
            for category in categories:
                cat_dir = os.path.join(bg_dir, category)
                if not os.path.exists(cat_dir):
                    os.makedirs(cat_dir)
        
        # Load backgrounds
        self.load_backgrounds()
    
    def load_backgrounds(self):
        """Load backgrounds from the backgrounds directory."""
        self.backgrounds = []
        self.categories = set(["All"])
        
        # Search for image files
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        
        for ext in extensions:
            for bg_path in glob.glob(os.path.join(self.bg_dir, "**", ext), recursive=True):
                # Get relative path to determine category
                rel_path = os.path.relpath(bg_path, self.bg_dir)
                parts = os.path.split(rel_path)
                
                # If in a subdirectory, use that as the category
                category = "General"
                if len(parts) > 1 and parts[0] != ".":
                    category = parts[0]
                    
                self.categories.add(category)
                
                try:
                    # Load the background image
                    bg_image = cv2.imread(bg_path)
                    
                    # Create thumbnail (keeping aspect ratio)
                    max_thumb_size = 120
                    h, w = bg_image.shape[:2]
                    aspect = w / h
                    
                    if w > h:
                        thumb_width = max_thumb_size
                        thumb_height = int(max_thumb_size / aspect)
                    else:
                        thumb_height = max_thumb_size
                        thumb_width = int(max_thumb_size * aspect)
                    
                    thumbnail = cv2.resize(bg_image, (thumb_width, thumb_height))
                    
                    # Store background info
                    self.backgrounds.append({
                        "path": bg_path,
                        "category": category,
                        "image": bg_image,
                        "thumbnail": thumbnail,
                        "width": w,
                        "height": h
                    })
                    
                except Exception as e:
                    print(f"Error loading background {bg_path}: {e}")
        
        # Add a special "None" option
        none_bg = np.zeros((100, 100, 3), dtype=np.uint8)
        none_bg.fill(50)
        
        # Draw a "no background" symbol
        cv2.line(none_bg, (30, 30), (70, 70), (200, 50, 50), 2)
        cv2.line(none_bg, (70, 30), (30, 70), (200, 50, 50), 2)
        
        self.backgrounds.insert(0, {
            "path": "none",
            "category": "All",
            "image": none_bg,
            "thumbnail": none_bg,
            "width": 100,
            "height": 100,
            "is_none": True
        })
        
        print(f"Loaded {len(self.backgrounds)} backgrounds in {len(self.categories)} categories")
    
    def get_filtered_backgrounds(self):
        """Get backgrounds filtered by the active category."""
        if self.active_category == "All":
            return self.backgrounds
        else:
            return [b for b in self.backgrounds if b["category"] == self.active_category or b.get("is_none", False)]
    
    def get_current_page_backgrounds(self):
        """Get backgrounds for the current page."""
        filtered = self.get_filtered_backgrounds()
        start_idx = self.current_page * self.items_per_page
        end_idx = min(start_idx + self.items_per_page, len(filtered))
        
        return filtered[start_idx:end_idx]
    
    def get_page_count(self):
        """Get the total number of pages."""
        filtered = self.get_filtered_backgrounds()
        return math.ceil(len(filtered) / self.items_per_page)
    
    def next_page(self):
        """Go to the next page."""
        max_page = self.get_page_count() - 1
        self.current_page = min(self.current_page + 1, max_page)
    
    def prev_page(self):
        """Go to the previous page."""
        self.current_page = max(0, self.current_page - 1)
    
    def set_active_category(self, category):
        """Set the active category and reset page."""
        if category in self.categories:
            self.active_category = category
            self.current_page = 0
    
    def get_background_by_index(self, index):
        """Get a background by its index on the current page."""
        backgrounds = self.get_current_page_backgrounds()
        if 0 <= index < len(backgrounds):
            return backgrounds[index]
        return None
    
    def set_background(self, background):
        """Set the current background."""
        self.current_background = background
    
    def clear_background(self):
        """Clear the current background."""
        self.current_background = None
    
    def set_opacity(self, opacity):
        """Set the background opacity."""
        self.background_opacity = max(0.0, min(1.0, opacity))
    
    def apply_background(self, canvas):
        """Apply the current background to the canvas."""
        if self.current_background is None or self.current_background.get("is_none", False):
            return canvas
        
        # Get background image
        bg_image = self.current_background["image"]
        
        # Resize background to match canvas
        h, w = canvas.shape[:2]
        bg_resized = cv2.resize(bg_image, (w, h))
        
        # Create a new blank canvas with alpha channel
        result = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Copy background to the RGB channels
        result[:, :, :3] = bg_resized
        
        # Set alpha channel based on opacity
        result[:, :, 3] = int(255 * self.background_opacity)
        
        # Composite the original canvas over the background
        if canvas.shape[2] == 4:  # If canvas has alpha channel
            alpha = canvas[:, :, 3] / 255.0
            alpha = np.stack([alpha, alpha, alpha], axis=2)
            
            # Blend RGB channels
            result[:, :, :3] = bg_resized * (1 - alpha) + canvas[:, :, :3] * alpha
            
            # Update alpha channel
            # New alpha = bg_alpha * (1 - canvas_alpha) + canvas_alpha
            bg_alpha = self.background_opacity * (1 - alpha[:, :, 0])
            new_alpha = bg_alpha + alpha[:, :, 0]
            result[:, :, 3] = (new_alpha * 255).astype(np.uint8)
        
        return result
    
    def draw_background_panel(self, image, panel_x, panel_y, panel_width, panel_height):
        """Draw the background panel on the image."""
        # Create panel background
        cv2.rectangle(image, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (40, 40, 50), -1)
        cv2.rectangle(image, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (70, 70, 90), 2)
        
        # Add title
        cv2.putText(image, "Backgrounds", (panel_x + 10, panel_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Add opacity slider
        slider_y = panel_y + 40
        slider_width = panel_width - 150
        
        cv2.putText(image, "Opacity:", (panel_x + 10, slider_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Draw slider track
        slider_x = panel_x + 90
        cv2.rectangle(image, (slider_x, slider_y), 
                     (slider_x + slider_width, slider_y + 10), 
                     (70, 70, 70), -1)
        
        # Calculate handle position
        handle_x = int(slider_x + slider_width * self.background_opacity)
        
        # Draw filled part
        cv2.rectangle(image, (slider_x, slider_y), 
                     (handle_x, slider_y + 10), 
                     (100, 100, 180), -1)
        
        # Draw handle
        cv2.circle(image, (handle_x, slider_y + 5), 6, (200, 200, 200), -1)
        cv2.circle(image, (handle_x, slider_y + 5), 6, (70, 70, 70), 1)
        
        # Show opacity value
        value_text = f"{int(self.background_opacity * 100)}%"
        cv2.putText(image, value_text, (slider_x + slider_width + 10, slider_y + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Store slider region
        slider_region = ((slider_x, slider_y - 5), (slider_x + slider_width, slider_y + 15))
        
        # Add category selector
        category_y = slider_y + 30
        cv2.putText(image, "Category:", (panel_x + 10, category_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Draw categories as dropdown
        dropdown_x = panel_x + 90
        dropdown_width = 150
        dropdown_height = 25
        
        cv2.rectangle(image, (dropdown_x, category_y - 5), 
                     (dropdown_x + dropdown_width, category_y + dropdown_height - 5), 
                     (60, 60, 80), -1)
        cv2.rectangle(image, (dropdown_x, category_y - 5), 
                     (dropdown_x + dropdown_width, category_y + dropdown_height - 5), 
                     (100, 100, 120), 1)
        
        # Show current category
        cv2.putText(image, self.active_category, (dropdown_x + 10, category_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        
        # Draw dropdown arrow
        arrow_x = dropdown_x + dropdown_width - 20
        arrow_y = category_y + 10
        cv2.line(image, (arrow_x, arrow_y), (arrow_x + 10, arrow_y), (200, 200, 200), 1)
        cv2.line(image, (arrow_x, arrow_y), (arrow_x + 5, arrow_y + 5), (200, 200, 200), 1)
        cv2.line(image, (arrow_x + 10, arrow_y), (arrow_x + 5, arrow_y + 5), (200, 200, 200), 1)
        
        # Store dropdown region
        dropdown_region = ((dropdown_x, category_y - 5), 
                          (dropdown_x + dropdown_width, category_y + dropdown_height - 5))
        
        # Draw backgrounds grid
        grid_y = category_y + dropdown_height + 15
        grid_height = panel_height - (grid_y - panel_y) - 40  # Leave space for navigation
        
        # Calculate grid layout
        thumb_size = 120
        thumb_margin = 15
        cols = (panel_width - 20) // (thumb_size + thumb_margin)
        rows = grid_height // (thumb_size + thumb_margin + 20)  # Extra space for labels
        
        # Get backgrounds for current page
        backgrounds = self.get_current_page_backgrounds()
        background_regions = []
        
        for i, bg in enumerate(backgrounds):
            row = i // cols
            col = i % cols
            
            if row >= rows:
                break
            
            thumb_x = panel_x + 10 + (thumb_size + thumb_margin) * col
            thumb_y = grid_y + (thumb_size + thumb_margin + 20) * row
            
            # Check if this is the current background
            is_selected = (self.current_background is not None and 
                          bg["path"] == self.current_background["path"])
            
            # Draw thumbnail background
            bg_color = (60, 60, 100) if is_selected else (30, 30, 40)
            border_color = (100, 100, 255) if is_selected else (70, 70, 90)
            
            cv2.rectangle(image, (thumb_x, thumb_y), 
                         (thumb_x + thumb_size, thumb_y + thumb_size), 
                         bg_color, -1)
            cv2.rectangle(image, (thumb_x, thumb_y), 
                         (thumb_x + thumb_size, thumb_y + thumb_size), 
                         border_color, 2)
            
            # Draw thumbnail
            thumbnail = bg["thumbnail"]
            
            # Center thumbnail in the square
            offset_x = (thumb_size - thumbnail.shape[1]) // 2
            offset_y = (thumb_size - thumbnail.shape[0]) // 2
            
            # Place thumbnail
            region_x = thumb_x + offset_x
            region_y = thumb_y + offset_y
            region_width = min(thumbnail.shape[1], thumb_size - offset_x)
            region_height = min(thumbnail.shape[0], thumb_size - offset_y)
            
            if region_width > 0 and region_height > 0:
                image[region_y:region_y+region_height, 
                     region_x:region_x+region_width] = thumbnail[:region_height, :region_width]
            
            # Add label
            if bg.get("is_none", False):
                label = "No background"
            else:
                # Get filename without extension as label
                label = os.path.splitext(os.path.basename(bg["path"]))[0]
                label = label[:12] + "..." if len(label) > 15 else label
            
            cv2.putText(image, label, (thumb_x, thumb_y + thumb_size + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Store background region
            background_regions.append({
                "index": i,
                "region": ((thumb_x, thumb_y), (thumb_x + thumb_size, thumb_y + thumb_size))
            })
        
        # Draw navigation controls
        nav_y = panel_y + panel_height - 30
        
        # Page indicator
        page_count = self.get_page_count()
        page_text = f"Page {self.current_page + 1} of {page_count}"
        
        cv2.putText(image, page_text, (panel_x + panel_width // 2 - 40, nav_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Previous page button
        prev_x = panel_x + 20
        prev_enabled = self.current_page > 0
        prev_color = (60, 60, 100) if prev_enabled else (50, 50, 70)
        
        cv2.rectangle(image, (prev_x, nav_y - 10), (prev_x + 80, nav_y + 15), 
                     prev_color, -1)
        cv2.putText(image, "< Prev", (prev_x + 15, nav_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Next page button
        next_x = panel_x + panel_width - 100
        next_enabled = self.current_page < page_count - 1
        next_color = (60, 60, 100) if next_enabled else (50, 50, 70)
        
        cv2.rectangle(image, (next_x, nav_y - 10), (next_x + 80, nav_y + 15), 
                     next_color, -1)
        cv2.putText(image, "Next >", (next_x + 15, nav_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Store navigation regions
        nav_regions = [
            {
                "action": "prev_page",
                "region": ((prev_x, nav_y - 10), (prev_x + 80, nav_y + 15)),
                "enabled": prev_enabled
            },
            {
                "action": "next_page",
                "region": ((next_x, nav_y - 10), (next_x + 80, nav_y + 15)),
                "enabled": next_enabled
            }
        ]
        
        return image, slider_region, dropdown_region, background_regions, nav_regions

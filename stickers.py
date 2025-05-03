import cv2
import numpy as np
import os
import glob
import math

class StickerManager:
    """Manages stickers for the drawing app."""
    
    def __init__(self, sticker_dir="stickers"):
        self.sticker_dir = sticker_dir
        self.stickers = []
        self.categories = set()
        self.active_category = "All"
        self.current_page = 0
        self.items_per_page = 12
        
        # Create stickers directory if it doesn't exist
        if not os.path.exists(sticker_dir):
            os.makedirs(sticker_dir)
            print(f"Created stickers directory: {sticker_dir}")
            
            # Create some example categories
            categories = ["shapes", "emoji", "animals", "symbols"]
            for category in categories:
                cat_dir = os.path.join(sticker_dir, category)
                if not os.path.exists(cat_dir):
                    os.makedirs(cat_dir)
        
        # Load stickers
        self.load_stickers()
    
    def load_stickers(self):
        """Load stickers from the stickers directory."""
        self.stickers = []
        self.categories = set(["All"])
        
        # Search for PNG files in all subdirectories
        for sticker_path in glob.glob(os.path.join(self.sticker_dir, "**", "*.png"), recursive=True):
            # Get relative path to determine category
            rel_path = os.path.relpath(sticker_path, self.sticker_dir)
            parts = os.path.split(rel_path)
            
            # If in a subdirectory, use that as the category
            category = "General"
            if len(parts) > 1 and parts[0] != ".":
                category = parts[0]
                
            self.categories.add(category)
            
            try:
                # Load the sticker image
                sticker = cv2.imread(sticker_path, cv2.IMREAD_UNCHANGED)
                
                # If sticker doesn't have alpha channel, add one
                if sticker.shape[2] == 3:
                    # Add alpha channel (fully opaque)
                    alpha = np.full((sticker.shape[0], sticker.shape[1]), 255, dtype=np.uint8)
                    sticker = cv2.merge([sticker[:, :, 0], sticker[:, :, 1], sticker[:, :, 2], alpha])
                
                # Create thumbnail
                max_thumb_size = 100
                original_size = max(sticker.shape[0], sticker.shape[1])
                scale = max_thumb_size / original_size
                
                thumb_width = int(sticker.shape[1] * scale)
                thumb_height = int(sticker.shape[0] * scale)
                
                thumbnail = cv2.resize(sticker, (thumb_width, thumb_height))
                
                # Store sticker info
                self.stickers.append({
                    "path": sticker_path,
                    "category": category,
                    "image": sticker,
                    "thumbnail": thumbnail,
                    "width": sticker.shape[1],
                    "height": sticker.shape[0]
                })
                
            except Exception as e:
                print(f"Error loading sticker {sticker_path}: {e}")
        
        print(f"Loaded {len(self.stickers)} stickers in {len(self.categories)} categories")
    
    def get_filtered_stickers(self):
        """Get stickers filtered by the active category."""
        if self.active_category == "All":
            return self.stickers
        else:
            return [s for s in self.stickers if s["category"] == self.active_category]
    
    def get_current_page_stickers(self):
        """Get stickers for the current page."""
        filtered = self.get_filtered_stickers()
        start_idx = self.current_page * self.items_per_page
        end_idx = min(start_idx + self.items_per_page, len(filtered))
        
        return filtered[start_idx:end_idx]
    
    def get_page_count(self):
        """Get the total number of pages."""
        filtered = self.get_filtered_stickers()
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
    
    def get_sticker_by_index(self, index):
        """Get a sticker by its index on the current page."""
        stickers = self.get_current_page_stickers()
        if 0 <= index < len(stickers):
            return stickers[index]
        return None
    
    def add_sticker_to_layer(self, layer, sticker, position, scale=1.0, rotation=0):
        """Add a sticker to a layer at the specified position with scaling and rotation."""
        if layer is None or sticker is None:
            return layer
        
        # Get sticker image
        img = sticker["image"]
        
        # Apply scaling
        if scale != 1.0:
            new_width = int(img.shape[1] * scale)
            new_height = int(img.shape[0] * scale)
            img = cv2.resize(img, (new_width, new_height))
        
        # Apply rotation if needed
        if rotation != 0:
            # Get rotation matrix
            center = (img.shape[1] // 2, img.shape[0] // 2)
            rot_mat = cv2.getRotationMatrix2D(center, rotation, 1.0)
            
            # Apply rotation
            img = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]), 
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
        
        # Calculate position to center the sticker
        x, y = position
        x = int(x - img.shape[1] / 2)
        y = int(y - img.shape[0] / 2)
        
        # Make sure coordinates are within the layer
        if x < 0 or y < 0 or x + img.shape[1] > layer.shape[1] or y + img.shape[0] > layer.shape[0]:
            # Adjust placement to fit within layer
            x = max(0, min(x, layer.shape[1] - img.shape[1]))
            y = max(0, min(y, layer.shape[0] - img.shape[0]))
            
            # If sticker is too large, skip it
            if x + img.shape[1] <= 0 or y + img.shape[0] <= 0:
                return layer
        
        # Calculate the region of the layer where the sticker will be placed
        layer_region = layer[y:y+img.shape[0], x:x+img.shape[1]]
        
        # Handle case where sticker extends beyond layer boundaries
        height_overlap = min(layer.shape[0] - y, img.shape[0])
        width_overlap = min(layer.shape[1] - x, img.shape[1])
        
        if height_overlap <= 0 or width_overlap <= 0:
            return layer
        
        sticker_region = img[:height_overlap, :width_overlap]
        layer_region = layer[y:y+height_overlap, x:x+width_overlap]
        
        # Extract alpha channel
        alpha = sticker_region[:, :, 3] / 255.0
        alpha = np.stack([alpha, alpha, alpha], axis=2)
        
        # Blend sticker with layer
        layer_region_new = layer_region.copy()
        
        # Blend RGB channels
        layer_region_new[:, :, :3] = (
            layer_region[:, :, :3] * (1 - alpha) + sticker_region[:, :, :3] * alpha
        )
        
        # Blend alpha channel
        layer_region_alpha = layer_region[:, :, 3] / 255.0
        sticker_alpha = sticker_region[:, :, 3] / 255.0
        
        # New alpha = current alpha + (sticker alpha * (1 - current alpha))
        new_alpha = layer_region_alpha + (sticker_alpha * (1 - layer_region_alpha))
        layer_region_new[:, :, 3] = (new_alpha * 255).astype(np.uint8)
        
        # Update the layer region
        layer[y:y+height_overlap, x:x+width_overlap] = layer_region_new
        
        return layer
    
    def draw_sticker_panel(self, image, panel_x, panel_y, panel_width, panel_height):
        """Draw the sticker panel on the image."""
        # Create panel background
        cv2.rectangle(image, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (40, 40, 50), -1)
        cv2.rectangle(image, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (70, 70, 90), 2)
        
        # Add title
        cv2.putText(image, "Stickers", (panel_x + 10, panel_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Add category selector
        category_y = panel_y + 40
        cv2.putText(image, "Category:", (panel_x + 10, category_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Draw categories as tabs
        categories = list(self.categories)
        categories.sort()
        if "All" in categories:
            # Move "All" to the front
            categories.remove("All")
            categories.insert(0, "All")
        
        tab_width = 80
        tab_height = 25
        tab_x = panel_x + 100
        tab_regions = []
        
        for i, category in enumerate(categories[:5]):  # Show at most 5 categories
            is_active = category == self.active_category
            
            # Draw tab
            color = (60, 60, 100) if is_active else (50, 50, 70)
            cv2.rectangle(image, (tab_x, category_y - 20), 
                         (tab_x + tab_width, category_y + tab_height - 20), 
                         color, -1)
            
            # Draw text
            text_color = (255, 255, 255) if is_active else (200, 200, 200)
            cv2.putText(image, category, (tab_x + 10, category_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            
            # Store tab region
            tab_regions.append({
                "category": category,
                "region": ((tab_x, category_y - 20), (tab_x + tab_width, category_y + tab_height - 20))
            })
            
            tab_x += tab_width + 5
        
        # Draw stickers grid
        grid_y = category_y + tab_height + 10
        grid_height = panel_height - (grid_y - panel_y) - 40  # Leave space for navigation
        
        # Calculate grid layout
        thumb_size = 80
        thumb_margin = 10
        cols = panel_width // (thumb_size + thumb_margin)
        rows = grid_height // (thumb_size + thumb_margin)
        
        # Get stickers for current page
        stickers = self.get_current_page_stickers()
        sticker_regions = []
        
        for i, sticker in enumerate(stickers):
            row = i // cols
            col = i % cols
            
            if row >= rows:
                break
            
            thumb_x = panel_x + (thumb_size + thumb_margin) * col + thumb_margin
            thumb_y = grid_y + (thumb_size + thumb_margin) * row
            
            # Draw thumbnail background
            cv2.rectangle(image, (thumb_x, thumb_y), 
                         (thumb_x + thumb_size, thumb_y + thumb_size), 
                         (30, 30, 40), -1)
            cv2.rectangle(image, (thumb_x, thumb_y), 
                         (thumb_x + thumb_size, thumb_y + thumb_size), 
                         (70, 70, 90), 1)
            
            # Draw thumbnail
            thumbnail = sticker["thumbnail"]
            
            # Center thumbnail in the square
            offset_x = (thumb_size - thumbnail.shape[1]) // 2
            offset_y = (thumb_size - thumbnail.shape[0]) // 2
            
            # Draw thumbnail with alpha
            if thumbnail.shape[2] == 4:
                # Extract alpha channel
                alpha = thumbnail[:, :, 3] / 255.0
                alpha = np.stack([alpha, alpha, alpha], axis=2)
                
                # Get region where thumbnail will be placed
                region_x = thumb_x + offset_x
                region_y = thumb_y + offset_y
                region = image[region_y:region_y+thumbnail.shape[0], 
                              region_x:region_x+thumbnail.shape[1]]
                
                # Blend thumbnail with background
                region_new = region.copy()
                region_new = region * (1 - alpha) + thumbnail[:, :, :3] * alpha
                
                # Update region
                image[region_y:region_y+thumbnail.shape[0], 
                     region_x:region_x+thumbnail.shape[1]] = region_new
            
            # Store sticker region
            sticker_regions.append({
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
        
        return image, tab_regions, sticker_regions, nav_regions

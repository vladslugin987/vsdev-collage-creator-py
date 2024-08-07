import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES
from tkinter import filedialog, messagebox, colorchooser
from tkinter import ttk as tk_tt, StringVar
from tkinter.scrolledtext import ScrolledText
import ttkbootstrap as ttk  # Импортируем ttkbootstrap для применения тем
import cv2
import os
import random
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont, ImageTk
import matplotlib.font_manager as fm  # For getting system fonts
import json
from concurrent.futures import ThreadPoolExecutor
import sys  # Для перенаправления stdout и stderr

class CollageCreator:
    def __init__(self):
        self.font_name = "Arial"
        self.font_path = fm.findSystemFonts(fontext='ttf')[0]  # Default to first system font
        self.font_size = 64
        self.text_color = (255, 255, 255)
        self.border_size = 2
        self.border_color = (0, 0, 0)

        self.xml_file_path = ""

        script_dir = os.path.dirname(__file__)
        self.face_detector = cv2.CascadeClassifier()
        if self.xml_file_path:
            self.load_cascade(self.xml_file_path)
        else:
            default_cascade_path = os.path.join(script_dir, 'haarcascade_frontalface_default.xml')
            self.load_cascade(default_cascade_path)

        self.draw_face_rectangle = False
        self.detect_faces = True  # Новая переменная для включения/выключения детекта лица
        self.smart_positioning = False  # Новая переменная для включения/выключения smart positioning
        self.draw_text_on_collage = True
        self.scale_factor = 1.05
        self.min_neighbors = 4
        self.total_images = 0
        self.progress_callback = None

        self.smart_offset_x = -320  # Добавляем атрибут smart_offset_x
        self.smart_offset_y = -320  # Добавляем атрибут smart_offset_y

    def load_cascade(self, cascade_path):
        if not self.face_detector.load(cascade_path):
            raise IOError(f"Error loading cascade file: {cascade_path}")

    def create_collage(self, images, folder_name):
        collage_width = 1000
        collage_height = 1000
        collage = np.full((collage_width, collage_height, 3), 255, dtype=np.uint8)
        positions = self.get_pattern(np.random.randint(0, 4))

        for i, (image, position) in enumerate(zip(images, positions)):
            face_rect = self.detect_face(image) if self.detect_faces else None
            square_image = self.crop_to_square(image, face_rect)
            resized_image = self.resize_image(square_image, position[2], position[3])
            self.draw_image(collage, resized_image, position[0], position[1])
            if self.draw_face_rectangle and face_rect is not None:
                self.draw_face_rectangle_on_collage(collage, position[0], position[1], face_rect, square_image, resized_image)

        if self.draw_text_on_collage:
            print(f"Drawing folder name: {folder_name}")
            self.draw_folder_name(collage, folder_name)
        else:
            print("draw_text_on_collage is disabled")
        return collage

    def get_pattern(self, index):
        patterns = [
            [(10, 10, 320, 320), (340, 10, 320, 320), (670, 10, 320, 320), (10, 340, 320, 320), (10, 670, 320, 320), (340, 340, 650, 650)],
            [(10, 10, 320, 320), (340, 10, 320, 320), (670, 10, 320, 320), (670, 340, 320, 320), (670, 670, 320, 320), (10, 340, 650, 650)],
            [(10, 10, 650, 650), (670, 10, 320, 320), (670, 340, 320, 320), (10, 670, 320, 320), (340, 670, 320, 320), (670, 670, 320, 320)],
            [(10, 10, 320, 320), (340, 10, 650, 650), (10, 340, 320, 320), (10, 670, 320, 320), (340, 670, 320, 320), (670, 670, 320, 320)],
        ]
        return patterns[index]

    def detect_face(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray_image, self.scale_factor, self.min_neighbors)
        if len(faces) > 0:
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            return faces[0]
        return None

    def crop_to_square(self, image, face_rect):
        h, w, _ = image.shape
        dim = min(w, h)
        if face_rect is not None:
            x, y, fw, fh = face_rect
            cx, cy = x + fw // 2, y + fh // 2
            x_offset = max(0, min(cx - dim // 2, w - dim))
            y_offset = max(0, min(cy - dim // 2, h - dim))
        else:
            if self.smart_positioning:
                x_offset = (w - dim) // 2 + self.smart_offset_x
                y_offset = (h - dim) // 2 + self.smart_offset_y
                x_offset = max(0, min(x_offset, w - dim))
                y_offset = max(0, min(y_offset, h - dim))
            else:
                x_offset = (w - dim) // 2
                y_offset = (h - dim) // 2
        cropped_image = image[y_offset:y_offset + dim, x_offset:x_offset + dim]
        if self.progress_callback:
            self.progress_callback()
        return cropped_image

    def resize_image(self, image, target_width, target_height):
        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

    def draw_image(self, collage, image, start_x, start_y):
        h, w, _ = image.shape
        collage[start_y:start_y + h, start_x:start_x + w] = image

    def draw_face_rectangle_on_collage(self, collage, start_x, start_y, face_rect, square_image, resized_image):
        fx, fy, fw, fh = face_rect
        scale_x = resized_image.shape[1] / square_image.shape[1]
        scale_y = resized_image.shape[0] / square_image.shape[0]
        rect_x = int(fx * scale_x) + start_x
        rect_y = int(fy * scale_y) + start_y
        rect_w = int(fw * scale_x)
        rect_h = int(fh * scale_y)
        cv2.rectangle(collage, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 0, 255), 2)

    def draw_folder_name(self, collage, folder_name):
        h, w, _ = collage.shape
        pil_image = Image.fromarray(collage)
        draw = ImageDraw.Draw(pil_image)
        font_path = self.font_path if self.font_path else fm.findSystemFonts(fontext='ttf')[0]  # Ensure a font path is set

        try:
            font = ImageFont.truetype(font_path, self.font_size)
            print(f"Loaded font: {font_path}")
        except IOError:
            font = ImageFont.load_default()
            print("Failed to load font, using default font.")

        text_bbox = draw.textbbox((0, 0), folder_name, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        text_x = (w - text_width) // 2
        text_y = (h - text_height) // 2  # Center the text vertically

        print(f"Text position: ({text_x}, {text_y}), Text size: ({text_width}, {text_height})")

        if self.border_size > 0:
            for adj_x in range(-self.border_size, self.border_size + 1):
                for adj_y in range(-self.border_size, self.border_size + 1):
                    if adj_x != 0 or adj_y != 0:
                        draw.text((text_x + adj_x, text_y + adj_y), folder_name, font=font, fill=self.border_color)

        draw.text((text_x, text_y), folder_name, font=font, fill=self.text_color)
        collage[:] = np.array(pil_image)
        print(f"Folder name '{folder_name}' drawn on collage")

    def save_image(self, image, file_path):
        cv2.imwrite(file_path, image)

    def save_settings(self, settings, file_path):
        # Преобразование значений цвета из кортежа в список
        settings['text_color'] = list(settings['text_color'])
        settings['border_color'] = list(settings['border_color'])
        with open(file_path, 'w') as file:
            json.dump(settings, file)
        print(f"Settings saved to {file_path}: {settings}")

    def load_settings(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                settings = json.load(file)
            # Преобразование значений цвета из списка в кортеж
            self.font_name = settings.get('font_name', self.font_name)
            self.font_size = settings.get('font_size', self.font_size)
            self.text_color = tuple(settings.get('text_color', self.text_color))
            self.border_size = settings.get('border_size', self.border_size)
            self.border_color = tuple(settings.get('border_color', self.border_color))
            self.draw_face_rectangle = settings.get('draw_face_rectangle', self.draw_face_rectangle)
            self.draw_text_on_collage = settings.get('draw_text_on_collage', self.draw_text_on_collage)
            self.scale_factor = settings.get('scale_factor', self.scale_factor)
            self.min_neighbors = settings.get('min_neighbors', self.min_neighbors)
            self.detect_faces = settings.get('detect_faces', self.detect_faces)
            self.smart_positioning = settings.get('smart_positioning', self.smart_positioning)
            self.xml_file_path = settings.get('xml_file_path', self.xml_file_path)
            if self.xml_file_path:
                self.load_cascade(self.xml_file_path)
        return None

class TextRedirector:
    def __init__(self, text_widget, tag):
        self.text_widget = text_widget
        self.tag = tag

    def write(self, message):
        self.text_widget.config(state="normal")
        self.text_widget.insert("end", message, (self.tag,))
        self.text_widget.config(state="disabled")
        self.text_widget.see("end")

    def flush(self):
        pass

class Application(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()

        self.style = ttk.Style("cosmo")  # Используем ttkbootstrap для применения темы

        self.title("VS Collage Creator by vsdev. / v2.0 (MacOS)")
        self.geometry("1000x800")
        self.minsize(1000, 800)
        self.collage_creator = CollageCreator()
        self.selected_directories = []
        self.collage_map = {}
        self.preview_frames = {}
        self.settings_file = "settings.json"
        self.image_count = 0

        self.load_settings()
        self.create_widgets()
        self.redirect_console_output()  # Добавляем перенаправление вывода консоли

    def create_widgets(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(2, weight=1)

        self.drag_and_drop_frame = ttk.Frame(self, height=100, width=260, relief="solid")
        self.drag_and_drop_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.drag_and_drop_label = ttk.Label(self.drag_and_drop_frame, text="Drag and drop the folders here")
        self.drag_and_drop_label.pack(expand=True, fill="both")

        self.drag_and_drop_frame.drop_target_register(DND_FILES)
        self.drag_and_drop_frame.dnd_bind('<<Drop>>', self.on_drop)

        self.log_text_area = ScrolledText(self, height=10, width=70, state="disabled")
        self.log_text_area.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.controls_frame = ttk.Frame(self)
        self.controls_frame.grid(row=1, column=0, columnspan=2, pady=10, sticky="nsew")

        self.save_all_button = ttk.Button(self.controls_frame, text="Save all collages", command=self.on_save_all_button_click)
        self.save_all_button.grid(row=0, column=0, padx=5)

        self.save_path_field = ttk.Entry(self.controls_frame)
        self.save_path_field.grid(row=0, column=1, padx=5, sticky="ew")

        self.browse_button = ttk.Button(self.controls_frame, text="Browse...", command=self.on_browse_button_click)
        self.browse_button.grid(row=0, column=2, padx=5)

        self.generate_button = ttk.Button(self.controls_frame, text="(Re)generate all", command=self.regenerate_all_collages)
        self.generate_button.grid(row=0, column=3, padx=5)

        self.clear_button = ttk.Button(self.controls_frame, text="Clear", command=self.on_clear_button_click)
        self.clear_button.grid(row=0, column=4, padx=5)

        self.settings_button = ttk.Button(self.controls_frame, text="Settings", command=self.on_settings_button_click)
        self.settings_button.grid(row=0, column=5, padx=5)

        self.controls_info_label = ttk.Label(self.controls_frame, text="[X] - Delete\n[R] - Regenerate\n[S] - Save", foreground="gray")
        self.controls_info_label.grid(row=0, column=6, padx=5)

        for i in range(7):
            self.controls_frame.grid_columnconfigure(i, weight=1)

        self.previews_canvas = tk.Canvas(self)
        self.previews_canvas.grid(row=2, column=0, columnspan=2, pady=10, sticky="nsew")

        self.previews_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.previews_canvas.yview)
        self.previews_scrollbar.grid(row=2, column=2, sticky="ns")

        self.previews_tile_pane = ttk.Frame(self.previews_canvas)

        self.previews_canvas.create_window((0, 0), window=self.previews_tile_pane, anchor="nw")
        self.previews_canvas.config(yscrollcommand=self.previews_scrollbar.set)

        self.previews_tile_pane.bind("<Configure>", lambda e: self.previews_canvas.configure(scrollregion=self.previews_canvas.bbox("all")))

        for i in range(5):
            self.previews_tile_pane.grid_columnconfigure(i, weight=1)

    def on_drop(self, event):
        if event.data:
            file_paths = self.tk.splitlist(event.data)
            for file_path in file_paths:
                if os.path.isdir(file_path):
                    self.selected_directories.append(file_path)
                    self.log(f"Folder added: {file_path}")
            self.update_drag_and_drop_label()

    def on_generate_button_click(self):
        for directory in self.selected_directories:
            self.load_images_from_selected_directory(directory)

    def on_save_all_button_click(self):
        save_path = self.save_path_field.get()
        if not save_path:
            save_path = filedialog.askdirectory()
            if save_path:
                self.save_path_field.insert(0, save_path)
            else:
                return

        if not os.path.isdir(save_path):
            messagebox.showerror("Error", "Invalid directory!")
            return

        for directory, collage in self.collage_map.items():
            folder_name = os.path.basename(directory)
            save_file = os.path.join(save_path, f"collage_{folder_name}.jpg")
            self.collage_creator.save_image(collage, save_file)
            self.log(f"Collage saved: {save_file}")

    def on_browse_button_click(self):
        save_path = filedialog.askdirectory()
        if save_path:
            self.save_path_field.delete(0, tk.END)
            self.save_path_field.insert(0, save_path)

    def on_clear_button_click(self):
        for widget in self.previews_tile_pane.winfo_children():
            widget.destroy()
        self.selected_directories.clear()
        self.drag_and_drop_label.config(text="Drag and drop the folders here")
        self.collage_map.clear()
        self.preview_frames.clear()
        self.log("Cleared all collages and reset selections.")

    def on_settings_button_click(self):
        settings_window = SettingsWindow(self)
        self.log("Settings window opened.")

    def update_drag_and_drop_label(self):
        text = "\n".join(self.selected_directories)
        self.drag_and_drop_label.config(text=text)

    def load_images_from_selected_directory(self, directory):
        self.log(f"Loading images from directory: {directory}")

        image_files = [os.path.join(directory, f) for f in os.listdir(directory) if
                       f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        self.log(f"Found {len(image_files)} images in {directory}")

        if len(image_files) < 6:
            self.log(f"Not enough images in {directory} to create a collage. Need at least 6 images.")
            return

        random.shuffle(image_files)
        image_files = image_files[:6]
        self.image_count = len(image_files)

        if not hasattr(self, 'loading_window') or not self.loading_window.winfo_exists():
            self.show_loading_window(self.image_count)
        else:
            self.update_loading_window_total(self.image_count)

        with ThreadPoolExecutor() as executor:
            images = list(executor.map(cv2.imread, image_files))

        self.log(f"Creating collage for {directory}")
        self.collage_creator.progress_callback = lambda: self.update_loading_progress(self.progress["value"] + 1)
        collage = self.collage_creator.create_collage(images, os.path.basename(directory))
        self.collage_map[directory] = collage
        self.display_preview(collage, os.path.basename(directory), directory)
        self.log(f"{datetime.now().strftime('%H:%M:%S')} - Collage generated for: {directory}")
        self.hide_loading_window()

    def display_preview(self, collage, folder_name, directory_path):
        preview_image = Image.fromarray(cv2.cvtColor(collage, cv2.COLOR_BGR2RGB))
        preview_image = preview_image.resize((150, 150), Image.LANCZOS)
        preview_photo = ImageTk.PhotoImage(preview_image)

        if directory_path in self.preview_frames:
            frame, preview_label, label = self.preview_frames[directory_path]
            preview_label.config(image=preview_photo)
            preview_label.image = preview_photo
            self.log(f"Updated preview for {directory_path}.")
        else:
            frame = ttk.Frame(self.previews_tile_pane)
            frame.grid(padx=10, pady=10, row=len(self.preview_frames) // 5, column=len(self.preview_frames) % 5)

            preview_label = ttk.Label(frame, image=preview_photo)
            preview_label.image = preview_photo
            preview_label.pack()
            preview_label.bind("<Button-1>", lambda e, d=directory_path: self.open_collage_window(d))

            label = ttk.Label(frame, text=folder_name)
            label.pack()

            button_frame = ttk.Frame(frame)
            button_frame.pack(pady=5)

            delete_button = ttk.Button(button_frame, text="X", command=lambda: self.delete_collage(directory_path), width=3)
            delete_button.pack(side="left", padx=5)

            regenerate_button = ttk.Button(button_frame, text="R", command=lambda: self.regenerate_collage(directory_path), width=3)
            regenerate_button.pack(side="left", padx=5)

            save_button = ttk.Button(button_frame, text="S", command=lambda: self.save_collage(directory_path), width=3)
            save_button.pack(side="left", padx=5)

            self.preview_frames[directory_path] = (frame, preview_label, label)
            self.log(f"Created new preview frame for {directory_path}.")

    def open_collage_window(self, directory_path):
        collage = self.collage_map[directory_path]
        folder_name = os.path.basename(directory_path)
        CollageWindow(self, collage, folder_name)

    def log(self, message):
        self.log_text_area.config(state="normal")
        self.log_text_area.insert("end", f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")
        self.log_text_area.config(state="disabled")
        self.log_text_area.see("end")

    def regenerate_all_collages(self):
        if not self.selected_directories:
            self.log("No directories selected for regeneration.")
            return
        self.image_count = 0
        for directory in self.selected_directories:
            self.image_count += len([f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        self.show_loading_window(self.image_count)
        self.log("Starting regeneration of all collages.")
        self.update_loading_progress(0)  # Set initial progress to 0%
        self.loading_window.update_idletasks()  # Force update the loading window to display it
        for i, directory in enumerate(self.selected_directories):
            self.log(f"Regenerating collage for directory: {directory}")
            self.regenerate_collage(directory)
        self.log("All collages regenerated.")
        self.hide_loading_window()

    def regenerate_collage(self, directory_path):
        self.log(f"Regenerating collage for {directory_path}")
        self.image_count = len([f for f in os.listdir(directory_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        if directory_path in self.collage_map:
            del self.collage_map[directory_path]  # Remove the old collage
        if directory_path in self.preview_frames:
            frame, preview_label, label = self.preview_frames[directory_path]
            frame.destroy()
            del self.preview_frames[directory_path]
        self.load_images_from_selected_directory(directory_path)  # Generate new collage
        self.display_preview(self.collage_map[directory_path], os.path.basename(directory_path), directory_path)
        self.update_collage_previews()  # Update previews after regeneration
        self.log(f"Collage regenerated for {directory_path}")

    def delete_collage(self, directory_path):
        self.log(f"Deleting collage for {directory_path}")
        if directory_path in self.collage_map:
            del self.collage_map[directory_path]
        if directory_path in self.preview_frames:
            frame, _, _ = self.preview_frames[directory_path]
            frame.destroy()
            del self.preview_frames[directory_path]

    def save_collage(self, directory_path):
        self.log(f"Saving collage for {directory_path}")
        if directory_path in self.collage_map:
            collage = self.collage_map[directory_path]
            save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")])
            if save_path:
                self.collage_creator.save_image(collage, save_path)
                self.log(f"Collage saved to {save_path}")

    def save_settings(self):
        settings = {
            'font_name': self.collage_creator.font_name,
            'font_size': self.collage_creator.font_size,
            'text_color': self.collage_creator.text_color,
            'border_size': self.collage_creator.border_size,
            'border_color': self.collage_creator.border_color,
            'draw_face_rectangle': self.collage_creator.draw_face_rectangle,
            'draw_text_on_collage': self.collage_creator.draw_text_on_collage,
            'scale_factor': self.collage_creator.scale_factor,
            'min_neighbors': self.collage_creator.min_neighbors,
            'detect_faces': self.collage_creator.detect_faces,
            'smart_positioning': self.collage_creator.smart_positioning,
            'smart_offset_x': self.collage_creator.smart_offset_x,
            'smart_offset_y': self.collage_creator.smart_offset_y,
            'xml_file_path': self.collage_creator.xml_file_path,
        }
        self.collage_creator.save_settings(settings, self.settings_file)

    def load_settings(self):
        settings = self.collage_creator.load_settings(self.settings_file)
        if settings:
            self.collage_creator.font_name = settings.get('font_name', self.collage_creator.font_name)
            self.collage_creator.font_size = settings.get('font_size', self.collage_creator.font_size)
            self.collage_creator.text_color = tuple(settings.get('text_color', self.collage_creator.text_color))
            self.collage_creator.border_size = settings.get('border_size', self.collage_creator.border_size)
            self.collage_creator.border_color = tuple(settings.get('border_color', self.collage_creator.border_color))
            self.collage_creator.draw_face_rectangle = settings.get('draw_face_rectangle',
                                                                    self.collage_creator.draw_face_rectangle)
            self.collage_creator.draw_text_on_collage = settings.get('draw_text_on_collage',
                                                                     self.collage_creator.draw_text_on_collage)
            self.collage_creator.scale_factor = settings.get('scale_factor', self.collage_creator.scale_factor)
            self.collage_creator.min_neighbors = settings.get('min_neighbors', self.collage_creator.min_neighbors)
            self.collage_creator.detect_faces = settings.get('detect_faces', self.collage_creator.detect_faces)
            self.collage_creator.smart_positioning = settings.get('smart_positioning',
                                                                  self.collage_creator.smart_positioning)
            self.collage_creator.smart_offset_x = settings.get('smart_offset_x', -320)
            self.collage_creator.smart_offset_y = settings.get('smart_offset_y', -320)
            self.collage_creator.xml_file_path = settings.get('xml_file_path', self.collage_creator.xml_file_path)
            if self.collage_creator.xml_file_path:
                self.collage_creator.load_cascade(self.collage_creator.xml_file_path)
        self.apply_settings()
        print(
            f"Loaded settings: smart_offset_x={self.collage_creator.smart_offset_x}, smart_offset_y={self.collage_creator.smart_offset_y}")

    def apply_settings(self):
        if hasattr(self, 'settings_window') and self.settings_window.winfo_exists():
            self.settings_window.update_color_pickers()

    def update_collage_previews(self):
        for i, (directory_path, (frame, preview_label, label)) in enumerate(self.preview_frames.items()):
            frame.grid(row=i // 5, column=i % 5, padx=10, pady=10)
        self.log("Updated collage previews.")

    def show_loading_window(self, total_images):
        if hasattr(self, 'loading_window') and self.loading_window.winfo_exists():
            return
        self.loading_window = tk.Toplevel(self)
        self.loading_window.title("Generating Collages")
        self.loading_window.geometry("300x100")
        self.loading_window.transient(self)
        self.loading_window.grab_set()

        self.center_window(self.loading_window, 300, 100)

        self.progress = ttk.Progressbar(self.loading_window, orient="horizontal", length=200, mode="determinate")
        self.progress.pack(pady=20)
        self.progress["maximum"] = total_images
        self.progress["value"] = 0

        self.progress_label = ttk.Label(self.loading_window, text="0%")
        self.progress_label.pack()

    def update_loading_window_total(self, total_images):
        if hasattr(self, 'progress'):
            self.progress["maximum"] = total_images

    def update_loading_progress(self, target_value):
        current_value = self.progress["value"]
        while current_value < target_value:
            current_value += 1
            self.progress["value"] = current_value
            percentage = (current_value / self.progress["maximum"]) * 100
            self.progress_label.config(text=f"{percentage:.0f}%")
            self.loading_window.update()
            self.loading_window.after(10)  # Задержка в 10 миллисекунд для плавного обновления

    def hide_loading_window(self):
        if hasattr(self, 'loading_window') and self.loading_window.winfo_exists():
            self.loading_window.destroy()

    def center_window(self, window, width, height):
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        window.geometry(f"{width}x{height}+{x}+{y}")

    def redirect_console_output(self):
        sys.stdout = TextRedirector(self.log_text_area, "stdout")
        sys.stderr = TextRedirector(self.log_text_area, "stderr")

class SettingsWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("Settings")
        self.geometry("600x700")
        self.collage_creator = parent.collage_creator
        parent.settings_window = self  # Сохраняем ссылку на окно настроек

        self.smart_offset_x = tk.IntVar(value="-320")
        self.smart_offset_y = tk.IntVar(value="-320")
        self.smart_offset_x_placeholder = "Default X offset"
        self.smart_offset_y_placeholder = "Default Y offset"

        self.create_widgets()

        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        ttk.Label(scrollable_frame, text="Select Font:").pack(pady=5)
        self.font_combo = ttk.Combobox(scrollable_frame, values=self.get_system_fonts())
        self.font_combo.set(self.collage_creator.font_name)
        self.font_combo.pack(pady=5)

        ttk.Label(scrollable_frame, text="Select Font Size:").pack(pady=5)
        self.font_size_spin = ttk.Spinbox(scrollable_frame, from_=10, to=100)
        self.font_size_spin.set(self.collage_creator.font_size)
        self.font_size_spin.pack(pady=5)

        ttk.Label(scrollable_frame, text="Select Text Color:").pack(pady=5)
        self.text_color_picker = tk.Button(scrollable_frame, bg=self.color_to_hex(self.collage_creator.text_color), width=20, command=self.choose_text_color)
        self.text_color_picker.pack(pady=5)

        ttk.Label(scrollable_frame, text="Select Border Size:").pack(pady=5)
        self.border_size_spin = ttk.Spinbox(scrollable_frame, from_=0, to=10)
        self.border_size_spin.set(self.collage_creator.border_size)
        self.border_size_spin.pack(pady=5)

        ttk.Label(scrollable_frame, text="Select Border Color:").pack(pady=5)
        self.border_color_picker = tk.Button(scrollable_frame, bg=self.color_to_hex(self.collage_creator.border_color), width=20, command=self.choose_border_color)
        self.border_color_picker.pack(pady=5)

        ttk.Label(scrollable_frame, text="Scale Factor:").pack(pady=5)
        self.scale_factor_spin = ttk.Spinbox(scrollable_frame, from_=1.0, to=2.0, increment=0.01)
        self.scale_factor_spin.set(self.collage_creator.scale_factor)
        self.scale_factor_spin.pack(pady=5)

        ttk.Label(scrollable_frame, text="Min Neighbors:").pack(pady=5)
        self.min_neighbors_spin = ttk.Spinbox(scrollable_frame, from_=1, to=10)
        self.min_neighbors_spin.set(self.collage_creator.min_neighbors)
        self.min_neighbors_spin.pack(pady=5)

        self.draw_face_rect_var = tk.BooleanVar(value=self.collage_creator.draw_face_rectangle)
        self.draw_face_rect_checkbox = ttk.Checkbutton(scrollable_frame, text="Draw Face Rectangle", variable=self.draw_face_rect_var)
        self.draw_face_rect_checkbox.pack(pady=5)

        self.draw_text_var = tk.BooleanVar(value=self.collage_creator.draw_text_on_collage)
        self.draw_text_checkbox = ttk.Checkbutton(scrollable_frame, text="Make text on the collages", variable=self.draw_text_var)
        self.draw_text_checkbox.pack(pady=5)

        self.detect_faces_var = tk.BooleanVar(value=self.collage_creator.detect_faces)
        self.detect_faces_checkbox = ttk.Checkbutton(scrollable_frame, text="Detect Faces", variable=self.detect_faces_var)
        self.detect_faces_checkbox.pack(pady=5)

        self.smart_positioning_var = tk.BooleanVar(value=self.collage_creator.smart_positioning)
        self.smart_positioning_checkbox = ttk.Checkbutton(scrollable_frame, text="Smart positioning (Beta)", variable=self.smart_positioning_var, command=self.toggle_smart_positioning)
        self.smart_positioning_checkbox.pack(pady=5)

        ttk.Label(scrollable_frame, text="Smart Positioning X Offset (px):").pack(pady=5)
        self.smart_offset_x_entry = ttk.Entry(scrollable_frame, textvariable=self.smart_offset_x)
        self.smart_offset_x_entry.pack(pady=5)

        ttk.Label(scrollable_frame, text="Smart Positioning Y Offset (px):").pack(pady=5)
        self.smart_offset_y_entry = ttk.Entry(scrollable_frame, textvariable=self.smart_offset_y)
        self.smart_offset_y_entry.pack(pady=5)

        ttk.Button(scrollable_frame, text="Select XML file for face detection", command=self.choose_xml_file).pack(pady=5)
        self.xml_file_label = ttk.Label(scrollable_frame, text="Current XML file: " + (self.collage_creator.xml_file_path if self.collage_creator.xml_file_path else "Default"))
        self.xml_file_label.pack(pady=5)

        ttk.Button(scrollable_frame, text="Save", command=self.save_settings).pack(pady=20)
        ttk.Button(scrollable_frame, text="About", command=self.open_about_window).pack(pady=20)

        self.update_color_pickers()  # Добавлено для обновления виджетов выбора цвета при открытии окна

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def get_system_fonts(self):
        font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
        font_names = [os.path.basename(fp).split('.')[0] for fp in font_paths]
        self.font_map = dict(zip(font_names, font_paths))
        return sorted(font_names)

    def choose_text_color(self):
        color = colorchooser.askcolor()[1]
        if color:
            self.text_color_picker.config(bg=color)

    def choose_border_color(self):
        color = colorchooser.askcolor()[1]
        if color:
            self.border_color_picker.config(bg=color)

    def choose_xml_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("XML files", "*.xml"), ("All files", "*.*")])
        if file_path:
            self.collage_creator.xml_file_path = file_path
            self.collage_creator.load_cascade(file_path)
            self.xml_file_label.config(text="Current XML file: " + file_path)

    def toggle_smart_positioning(self):
        if self.smart_positioning_var.get():
            self.smart_offset_x_entry.config(state="normal")
            self.smart_offset_y_entry.config(state="normal")
        else:
            self.smart_offset_x_entry.config(state="disabled")
            self.smart_offset_y_entry.config(state="disabled")

    def clear_placeholder_x(self, event):
        if self.smart_offset_x_entry.get() == self.smart_offset_x_placeholder:
            self.smart_offset_x_entry.delete(0, "end")
            self.smart_offset_x_entry.config(foreground='black')

    def add_placeholder_x(self, event):
        if not self.smart_offset_x_entry.get():
            self.smart_offset_x_entry.insert(0, self.smart_offset_x_placeholder)
            self.smart_offset_x_entry.config(foreground='grey')

    def clear_placeholder_y(self, event):
        if self.smart_offset_y_entry.get() == self.smart_offset_y_placeholder:
            self.smart_offset_y_entry.delete(0, "end")
            self.smart_offset_y_entry.config(foreground='black')

    def add_placeholder_y(self, event):
        if not self.smart_offset_y_entry.get():
            self.smart_offset_y_entry.insert(0, self.smart_offset_y_placeholder)
            self.smart_offset_y_entry.config(foreground='grey')

    def color_to_hex(self, color):
        return f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"

    def update_color_pickers(self):
        self.text_color_picker.config(bg=self.color_to_hex(self.collage_creator.text_color))
        self.border_color_picker.config(bg=self.color_to_hex(self.collage_creator.border_color))

    def save_settings(self):
        self.collage_creator.font_name = self.font_combo.get()
        self.collage_creator.font_path = self.font_map.get(self.collage_creator.font_name, None)
        self.collage_creator.font_size = int(self.font_size_spin.get())
        self.collage_creator.text_color = self.hex_to_color(self.text_color_picker.cget("bg"))
        self.collage_creator.border_size = int(self.border_size_spin.get())
        self.collage_creator.border_color = self.hex_to_color(self.border_color_picker.cget("bg"))
        self.collage_creator.scale_factor = float(self.scale_factor_spin.get())
        self.collage_creator.min_neighbors = int(self.min_neighbors_spin.get())
        self.collage_creator.draw_face_rectangle = self.draw_face_rect_var.get()
        self.collage_creator.draw_text_on_collage = self.draw_text_var.get()
        self.collage_creator.detect_faces = self.detect_faces_var.get()
        self.collage_creator.smart_positioning = self.smart_positioning_var.get()

        if self.smart_offset_x.get() == self.smart_offset_x_placeholder or not self.smart_offset_x.get():
            self.collage_creator.smart_offset_x = -320
        else:
            self.collage_creator.smart_offset_x = int(self.smart_offset_x.get())

        if self.smart_offset_y.get() == self.smart_offset_y_placeholder or not self.smart_offset_y.get():
            self.collage_creator.smart_offset_y = -320
        else:
            self.collage_creator.smart_offset_y = int(self.smart_offset_y.get())

        self.parent.save_settings()
        self.destroy()
        self.parent.log("Settings saved and applied.")
        print(
            f"Saved settings: smart_offset_x={self.collage_creator.smart_offset_x}, smart_offset_y={self.collage_creator.smart_offset_y}")

    def open_about_window(self):
        AboutWindow(self, self.parent)

    def hex_to_color(self, hex):
        return tuple(int(hex[i:i + 2], 16) for i in (1, 3, 5))

    def on_close(self):
        self.destroy()
        self.parent.log("Settings window closed.")


class AboutWindow(tk.Toplevel):
    def __init__(self, parent, master):
        super().__init__(parent)
        self.parent = master
        self.title("About")
        self.geometry("700x700")
        self.create_widgets()

        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        about_text = (
            "VS Collage Creator\n"
            "Version 2.0 (MacOS ARM)\n\n"
            "Developer: vsdev.\n\n\n"
            "Usage:\n\n"
            "Buttons:\n"
            "- Save all collages: Save all generated collages to the selected directory.\n"
            "- (Re)generate all: Regenerate all collages from the selected directories.\n"
            "- Clear: Clear all selected directories and previews.\n"
            "- Settings: Open the settings window to customize collage creation.\n\n"
            "Settings:\n"
            "- Font: Choose the font for text on the collage.\n"
            "- Font Size: Set the size of the font.\n"
            "- Text Color: Choose the color of the text.\n"
            "- Border Size: Set the size of the border around the text.\n"
            "- Border Color: Choose the color of the border around the text.\n"
            "- Scale Factor: Set the scale factor for face detection. \n  The higher the value, the smaller the faces that can be detected. \n  Lower values may improve detection speed but may miss smaller faces.\n"
            "- Min Neighbors: Set the minimum neighbors for face detection. \n  Lower values will detect more faces but may result in more false positives. \n  Higher values will reduce false positives but may miss some faces.\n"
            "- Draw Face Rectangle: Toggle drawing rectangles around detected faces.\n"
            "- Make text on the collages: Toggle drawing text on the collages.\n"
            "- Detect Faces: Toggle face detection.\n"
            "- Smart positioning (Beta): Toggle smart positioning of images. Allows setting X and Y offsets.\n"
            "- Select XML file for face detection: Choose an XML file for face detection.\n\n"
            "Default Settings:\n"
            "- Font: Arial\n"
            "- Font Size: 64\n"
            "- Text Color: White\n"
            "- Border Size: 2\n"
            "- Border Color: Black\n"
            "- Scale Factor: 1.05\n"
            "- Min Neighbors: 4\n"
            "- Detect Faces: On\n"
            "- Smart Positioning: Off\n"
        )

        ttk.Label(scrollable_frame, text=about_text, justify="left").pack(padx=10, pady=10, anchor="w")

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def on_close(self):
        self.destroy()
        self.parent.log("About window closed.")


class CollageWindow(tk.Toplevel):
    def __init__(self, parent, collage, folder_name):
        super().__init__(parent)
        self.title(f"Collage - {folder_name}")
        self.geometry("1200x1000")
        self.collage = collage

        self.create_widgets()

    def create_widgets(self):
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill="both", expand=True)

        self.collage_image = Image.fromarray(cv2.cvtColor(self.collage, cv2.COLOR_BGR2RGB))
        self.collage_photo = ImageTk.PhotoImage(self.collage_image)

        self.current_collage_view = ttk.Label(self.main_frame, image=self.collage_photo)
        self.current_collage_view.image = self.collage_photo
        self.current_collage_view.pack(side="top", fill="both", expand=True)

        self.back_button = ttk.Button(self.main_frame, text="Back", command=self.destroy)
        self.back_button.pack(side="bottom", pady=10)


if __name__ == "__main__":
    app = Application()
    app.mainloop()

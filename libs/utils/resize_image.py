from PIL import Image
import os

def resize_image(img_path, size=(800, 800)):
    """
    Resizes the given .png image to the specified size and saves it with '_resized' suffix.

    :param img_path: The path to the input image (must be a .png file).
    :param size: A tuple (width, height) representing the target size of the image.
    """
    try:
        # Open the image file
        with Image.open(img_path) as img:
            # Check if the image is in PNG format
            if img.format != 'PNG':
                raise ValueError("Input file is not a PNG image.")
            
            # Resize the image
            img_resized = img.resize(size)
            
            # Generate the new filename
            img_name, _ = os.path.splitext(img_path)
            new_filename = f"{img_name}_resized.png"
            
            # Save the resized image
            img_resized.save(new_filename)
            print(f"Image resized and saved as {new_filename}")
    
    except Exception as e:
        print(f"Error resizing image: {e}")

if __name__ == "__main__":
    # Test the function with a sample image
    img_path = "/data1/zhaofanghan/Advanced_ML_Coursework/report/figures/small_1.png"
    resize_image(img_path)
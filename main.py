import cv2
import numpy as np
import math

def pixelate_frame(frame, pixelation_size=1):
    height, width = frame.shape[:2]

    temp_frame = cv2.resize(frame, (width // pixelation_size, height // pixelation_size),
                            interpolation=cv2.INTER_LINEAR)
    pixelated_frame = cv2.resize(temp_frame, (width, height), interpolation=cv2.INTER_NEAREST)
    return pixelated_frame


def increase_saturation(frame, saturation_scale=1, green_tint_strength=0):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_frame)

    s = np.clip(s * saturation_scale, 0, 255).astype(np.uint8)

    saturated_frame = cv2.merge([h, s, v])
    saturated_frame = cv2.cvtColor(saturated_frame, cv2.COLOR_HSV2BGR)
    frame_float = saturated_frame.astype(np.float32)

    frame_float[..., 1] += green_tint_strength  # Increase green channel
    frame_float = np.clip(frame_float, 0, 255).astype(np.uint8)  # Clip values to [0, 255]

    return frame_float
    return saturated_frame


def hexagon_pixelate(frame, hex_size=50):
    height, width, _ = frame.shape
    hex_height = int(math.sqrt(3) * hex_size)

    output_frame = frame.copy()

    for row in range(0, height + hex_height, hex_height):
        for col in range(0, width + hex_size, int(1.5 * hex_size)):
            # Offset every second row for hexagonal tiling
            if (row // hex_height) % 2 == 0:
                x_offset = col
            else:
                x_offset = col + int(0.75 * hex_size)

            # Ensure hexagons stay within image boundaries
            if x_offset - hex_size < 0 or x_offset + hex_size > width or row - hex_height < 0 or row + hex_height > height:
                continue


            hex_points = get_hexagon_points(x_offset, row, hex_size)
            mask = np.zeros_like(frame, dtype=np.uint8)
            cv2.fillPoly(mask, [hex_points], (255, 255, 255))
            hex_region = cv2.bitwise_and(frame, mask)
            avg_color = cv2.mean(hex_region, mask=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))[:3]
            cv2.fillPoly(output_frame, [hex_points], avg_color)

    return output_frame


def get_hexagon_points(x_center, y_center, size):
    """
    Returns the 6 points of a regular hexagon centered at (x_center, y_center)
    """
    points = []
    for i in range(6):
        angle = 2 * np.pi / 6 * i  # Regular hexagon angles
        x_i = x_center + size * math.cos(angle)
        y_i = y_center + size * math.sin(angle)
        points.append([int(x_i), int(y_i)])
    return np.array(points, dtype=np.int32)

ASCII_CHARS = "@#W$9876543210?!abc;:+=-,._  '`"
#ASCII_CHARS = "@%#*+=-:. "

def map_brightness_to_ascii(average_brightness):
    num_chars = len(ASCII_CHARS)
    return ASCII_CHARS[int(average_brightness / 255 * (num_chars - 1))]


def ascii_art_to_image(frame, block_size=8, font_scale=0.4, font_thickness=1):
    gray_frame = 255 - cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray_frame.shape


    output_image_height = height // block_size * 12  # Approximate height for font
    output_image_width = width // block_size * 7  # Approximate width for font

    ascii_image = np.zeros((output_image_height, output_image_width), dtype=np.uint8) * 255

    # Set font type for ASCII art rendering
    font = cv2.FONT_HERSHEY_DUPLEX

    # Coordinates for placing text
    y_offset = 10  # Vertical offset to align text properly

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = gray_frame[y:y + block_size, x:x + block_size]

            # Compute the average brightness of the block
            average_brightness = np.mean(block)

            # Map the brightness to an ASCII character
            ascii_char = map_brightness_to_ascii(average_brightness)

            # Determine the position to draw the ASCII character
            text_x = (x // block_size) * 7  # Adjust horizontal offset
            text_y = (y // block_size) * 14 + y_offset  # Adjust vertical offset

            # Draw the ASCII character on the image
            cv2.putText(ascii_image, ascii_char, (text_x, text_y), font, font_scale, (255,), font_thickness)

    return ascii_image


def main():
    # Initialize video capture (0 is usually the default webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Variable to store the previous frame
    prev_frame = None
    alpha = 0  # Opacity of the previous frame (0.0 to 1.0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        # Flip the frame horizontally
        flipped_frame = cv2.flip(frame, 1)

        # Initialize prev_frame with the first frame
        if prev_frame is None:
            prev_frame = flipped_frame.copy()

        # Blend the current frame with the previous frame
        blended_frame = cv2.addWeighted(flipped_frame, 1 - alpha, prev_frame, alpha, 0)

        # Apply saturation filter
        saturated_frame = increase_saturation(blended_frame, saturation_scale=1)

        # Apply pixelation filter
        square_frame = pixelate_frame(saturated_frame, pixelation_size=7)

        cv2.imshow('Webcam', square_frame)

        # Update the previous frame to the current frame for the next iteration
        prev_frame = flipped_frame.copy()
        prev_frame = 255 - prev_frame

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()


def main2():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error")
        return
    prev_frame = None

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        # Flip the frame horizontally
        flipped_frame = cv2.flip(frame, 1)

        gray_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2GRAY)

        # Initialize prev_frame as the first grayscale frame
        if prev_frame is None:
            prev_frame = gray_frame.copy()
            continue
        gray_frame = pixelate_frame(gray_frame, pixelation_size=3)

        # Calculate the absolute difference between the current frame and the previous frame
        diff_frame = cv2.absdiff(gray_frame, prev_frame)

        # Apply a threshold to highlight significant differences (edges)
        _, thresh_frame = cv2.threshold(diff_frame, 20, 255, cv2.THRESH_BINARY)

        #Apply a blur to smooth the edges
        #blurred_frame = cv2.GaussianBlur(thresh_frame, (5, 5), 0)

        # Display the edge-detected frame
        cv2.imshow('Edge Detection via Motion', thresh_frame)

        # Update the previous frame to be the current frame for the next iteration
        prev_frame = gray_frame.copy()

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()


def main3():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        # Flip the frame horizontally (mirror effect)
        flipped_frame = cv2.flip(frame, 1)

        # Convert the frame to ASCII art and display it as an image
        ascii_image = ascii_art_to_image(flipped_frame, block_size=6)
        resized_ascii_image = cv2.resize(ascii_image, (frame.shape[1]+400, frame.shape[0]+200), interpolation=cv2.INTER_NEAREST)
        # Display the ASCII art image
        cv2.imshow('ASCII Art Webcam', resized_ascii_image)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

#main() runs the pixelation filter, main2() runs the edge detection, main3() runs the image to ascii
if __name__ == "__main__":
    main2()

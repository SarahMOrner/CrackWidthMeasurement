import cv2
import json
import numpy as np
import random

def draw_crack_outline(image, points):
    #Draw semi-transparent dark blue lines representing the overall crack outline and store them in crackOutline
    crackOutline = []
    for i in range(len(points)):
        start_point = tuple(map(int, points[i]))
        end_point = tuple(map(int, points[(i + 1) % len(points)]))
        cv2.line(image, start_point, end_point, (255, 0, 0), 16)
        crackOutline.append((start_point, end_point))  # Dark blue lines
    return image, crackOutline

def line_intersects_segment(p1, p2, q1, q2):
    # Check if two line segments (p1-p2 and q1-q2) intersect and return the intersection point
    def cross(v1, v2):
        return v1[0] * v2[1] - v1[1] * v2[0]

    r = (p2[0] - p1[0], p2[1] - p1[1])
    s = (q2[0] - q1[0], q2[1] - q1[1])
    rxs = cross(r, s)
    qp = (q1[0] - p1[0], q1[1] - p1[1])

    if rxs == 0:
        return None  # Lines are parallel
    t = cross(qp, s) / rxs
    u = cross(qp, r) / rxs
    if 0 <= t <= 1 and 0 <= u <= 1:
        return int(p1[0] + t * r[0]), int(p1[1] + t * r[1])
    return None

def extend_line_to_collision_or_bounds(p1, p2, image_shape, crack_outline):
    #Extend a line perpendicular to the orange line until it collides with the blue crack outline or reaches the image bounds
    # Exclude the segment if no collision occurs before the bound is reached
    height, width = image_shape[:2]

    # Calculate midpoint and direction vector for the perpendicular line
    mid_x = (p1[0] + p2[0]) // 2
    mid_y = (p1[1] + p2[1]) // 2
    dx = p2[1] - p1[1]  # Perpendicular direction x
    dy = -(p2[0] - p1[0])  # Perpendicular direction y
    magnitude = np.hypot(dx, dy)

    if magnitude == 0:
        return []  # Avoid division by zero
    dx = dx / magnitude * max(width, height)
    dy = dy / magnitude * max(width, height)

    # Extend line in both directions
    directions = [
        (mid_x + dx, mid_y + dy),  # Positive direction
        (mid_x - dx, mid_y - dy)   # Negative direction
    ]

    def clip_to_bounds(x, y):
        return max(0, min(int(x), width - 1)), max(0, min(int(y), height - 1))

    valid_segments = []
    for direction in directions:
        start = (mid_x, mid_y)
        end = clip_to_bounds(*direction)
        found_intersection = False
        for edge in crack_outline:
            intersection = line_intersects_segment(start, end, edge[0], edge[1])
            if intersection:
                valid_segments.append((start, intersection))
                found_intersection = True
                break
        if not found_intersection:
            continue  # Exclude segment if no intersection is found

    return valid_segments  # Return only valid red line segments

def draw_neighbor_connections(image, points, crackOutline):
    triangleEdges = []  # Initialize the list to store neighbor connections
    overlay = image.copy()  # Create overlay for triangle fills
    max_width = 0  # Variable to track the maximum width
    max_width_segment = None  # Track the widest segment

    for i in range(len(points)):
        current_point = tuple(map(int, points[i]))
        prev_point = tuple(map(int, points[(i - 1) % len(points)]))  # Previous neighbor
        next_point = tuple(map(int, points[(i + 1) % len(points)]))  # Next neighbor

        # Generate random color for this triangle
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # Fill triangles with the generated color
        triangle_pts = np.array([current_point, prev_point, next_point], np.int32)
        cv2.fillPoly(overlay, [triangle_pts], color)

        # Draw orange lines to neighbors
        cv2.line(image, current_point, prev_point, (0, 165, 255), 2)
        cv2.line(image, current_point, next_point, (0, 165, 255), 2)
        cv2.line(image, prev_point, next_point, (0, 165, 255), 3)

        # Extend perpendicular lines and color them with the triangle's color
        red_line_segments = extend_line_to_collision_or_bounds(prev_point, next_point, image.shape, crackOutline)
        for segment in red_line_segments:  # Draw valid segments
            length = np.linalg.norm(np.array(segment[0]) - np.array(segment[1]))
            if length > max_width:  # Track the maximum width
                max_width = length
                max_width_segment = segment
            else:
                cv2.line(image, segment[0], segment[1], color, 2)

    # Draw the widest width segment as an opaque line and label it
    if max_width_segment:
        cv2.line(image, max_width_segment[0], max_width_segment[1], (0, 0, 0), 4)
        midpoint = ((max_width_segment[0][0] + max_width_segment[1][0]) // 2, 
                    (max_width_segment[0][1] + max_width_segment[1][1]) // 2)
        cv2.putText(image, f"Max Width: {max_width:.2f}px", midpoint, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Blend the overlay back onto the image
    cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)

    print(f"Maximum crack width (perpendicular): {max_width:.2f} pixels")
    return image, max_width

def main(json_file, input_image_path, output_image_path):
    """Main function to process and visualize crack annotations."""
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Extract points from the JSON
    points = data["shapes"][0]["points"]
    points = [tuple(map(int, point)) for point in points]  # Convert to integer tuples

    # Load the base image
    base_image = cv2.imread(input_image_path)
    if base_image is None:
        raise FileNotFoundError(f"Image not found at {input_image_path}")

    # Copy the image to draw separately for each visualization
    dark_blue_image = base_image.copy()

    # Draw the dark blue outline (crack loop)
    dark_blue_image, crackOutline = draw_crack_outline(dark_blue_image, points)

    # Draw the orange individual connections with red lines and fill triangles
    final_image, max_width = draw_neighbor_connections(dark_blue_image, points, crackOutline)

    # Save the final output
    cv2.imwrite(output_image_path, final_image)
    print(f"Final visualization saved to {output_image_path}")




if __name__ == "__main__":
    json_file = r"C:\Users\sarah\OneDrive\Documents\ARA\crackExample\excrack30nodesann.json"
    input_image_path = r"C:\Users\sarah\OneDrive\Documents\ARA\crackExample\excrack30nodes.png"
    output_image_path = r"C:\Users\sarah\OneDrive\Documents\ARA\CalculateWidth/neighboringnodesoutputmores.jpg"

    main(json_file, input_image_path, output_image_path)

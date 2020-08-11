# ----------------------------------------------------------------------------------------------------------------------
#   Author: Sean Deely
#   ID:     1674836
#   Date:   28/04/20
# ----------------------------------------------------------------------------------------------------------------------

import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from math import sqrt


# ----------------------------------------------------------------------------------------------------------------------
def get_harris_response(image, sigma):
    """ Form the Harris response for each image so that the interest points can be found. """
    I_x = filters.gaussian_filter(image, sigma, order=(1, 0))
    I_y = filters.gaussian_filter(image, sigma, order=(0, 1))

    A = filters.gaussian_filter(I_x * I_x, sigma * 2.5)
    B = filters.gaussian_filter(I_x * I_y, sigma * 2.5)
    C = filters.gaussian_filter(I_y * I_y, sigma * 2.5)

    det_M = (A * C) - (B * B)
    tra_M = A + C

    return det_M / tra_M
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def get_harris_points(harris_im, threshold, min_d):
    """ Threshold the Harris responses and return co-ordinates of Harris interest points. """
    # Find the top corner candidates above a threshold
    corner_threshold = harris_im.max() * threshold
    harris_im_th = (harris_im > corner_threshold)

    # Find the co-ordinates of these candidates and their values
    coords = np.array(harris_im_th.nonzero()).T
    candidate_values = np.array([harris_im[c[0],c[1]] for c in coords])
    indicies = np.argsort(candidate_values)

    # Store allowed point locations in boolean image
    allowed_locations = np.zeros(harris_im.shape, dtype='bool')
    allowed_locations[min_d:-min_d, min_d:-min_d] = True

    # Select the best points, using nonmax suppression based on
    # the array of allowed locations
    filtered_coords = []
    for i in indicies[::-1]:
        r,c = coords[i]     # reversed inidices
        if allowed_locations[r,c]:
            filtered_coords.append((r,c))
            allowed_locations[r - min_d:r + min_d + 1, c - min_d:c + min_d + 1] = False
    return filtered_coords
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def plot_harris_points(image, filtered_coords):
    """ plot each of the Harris interest points over the image itslef. """
    plt.imshow(image, cmap="gray")
    y, x = np.transpose(filtered_coords)
    plt.plot(x, y, 'r.')
    plt.axis('off')
    # plt.show()
    plt.close()
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def get_descriptors(image, interest_points, width):
    """ 1. Form an image patch around the first interest point (11x11 pixels)
        2. Flatten it into a 121 element vector.
        3. Subtract the mean of that vector from the vector itself
        4. Normalize the vector so values range from -1 to +1.
        5. Repeat for all interest points, stacking the vectors each time """
    descriptors = []
    for point in interest_points:
        image_patch = image[point[0] - width: point[0] + width + 1, point[1] - width: point[1] + width + 1].flatten()
        image_patch -= np.mean(image_patch)
        descriptors.append(image_patch/np.linalg.norm(image_patch))
    return descriptors
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def plot_response_matrix(r12, r12_th):
    """ Plot the response matrix and the thresholded response matrix. """
    plt.imshow(r12, cmap='gray')
    # plt.show()
    plt.close()
    plt.imshow(r12_th, cmap='gray')
    # plt.show()
    plt.close()
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def get_response_matrix(m1, m2, threshold):
    """ Form the response matrix, R12, by transposing the M2 matrix and taking its dot product with matrix M1.
        Then threshold R12 so that only mappings > 0.95 (i.e. very strong mappings) get returned. """
    m1 = np.array(m1)
    m2 = np.array(m2)

    r12 = np.dot(m1, m2.T)
    r12_th = r12.copy()

    coords = []
    for r in range(r12.shape[0]):
        for c in range(r12.shape[1]):
            if r12[r][c] > threshold:
                coords.append((r,c))
            else: r12_th[r][c] = 0

    plot_response_matrix(r12, r12_th)

    return coords
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def plot_mapping(coord_pairs, im_1, im_2, interest_points_im_1, interest_points_im_2):
    """ Using the coordinate pairs obtained from the thresholded response matrix,
        plot the mappings between image 1 and image 2. If the images are of different size,
        pad the smaller image with zeros. """

    # Check if image dimensions are the same..
    if im_1.shape != im_2.shape:
        # if image 1 is smaller, pad image 1 with zeros
        if im_1.shape < im_2.shape:
            padded_im_1 = np.zeros(im_2.shape)
            padded_im_1[:im_1.shape[0], :im_1.shape[1]] = im_1
            combined_im = np.concatenate((padded_im_1, im_2), axis=1)
        # otherwise, pad image 2 with zeros
        else:
            padded_im_2 = np.zeros(im_1.shape)
            padded_im_2[:im_2.shape[0], :im_2.shape[1]] = im_2
            combined_im = np.concatenate((im_1, padded_im_2), axis=1)
    else: combined_im = np.concatenate((im_1, im_2), axis=1)

    plt.imshow(combined_im, cmap='gray')

    # N.B., interest_points are (y,x) or (r,c)..
    for i in range(np.array(coord_pairs).shape[0]):
        c1, c2 = coord_pairs[i]
        plt.plot([interest_points_im_1[c1][1], interest_points_im_2[c2][1] + im_1.shape[0]],
                 [interest_points_im_1[c1][0], interest_points_im_2[c2][0]], 'y', linewidth=1)
        plt.plot(interest_points_im_1[c1][1], interest_points_im_1[c1][0], 'ro', markersize=4)
        plt.plot(interest_points_im_2[c2][1] + im_1.shape[0],
                 interest_points_im_2[c2][0], 'bo', markersize=4)

    # For best view of mappings, use plt.show() as plt.savefig outputs are very small
    plt.axis('off')
    # plt.show()
    plt.close()
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def get_best_translation(im_1, mappings, ips_image_1, ips_image_2, threshold):
    """ Found the translation with the strongest mapping between image 1 and image 2. """

    # build a list of translations which consists of the differences in row and column coordinates
    # between the two images. These translations "should" all be the same distance apart and have
    # the same slope if the response matrix was correctly thresholded (0.95).
    # If this threshold was lower, the translations would not be the same due to outliers.
    translations = np.zeros((len(mappings), 2))
    for i in range(len(mappings)):
        r,c = mappings[i]
        translations[i][0] = ips_image_1[r][0] - ips_image_2[c][0]
        translations[i][1] = ips_image_1[r][1] - ips_image_2[c][1]

    largest_number_of_agreements = 0
    # Set best row and column translations to height of original image.
    # This is worst case scenario.
    best_r_translation = im_1.shape[0]
    best_c_translation = im_1.shape[0]

    # For each translation in the list....
    for i in range(len(mappings)):
        strong_agreements = 1
        weak_agreements = 0
        r_translation = translations[i][0]
        c_translation = translations[i][1]

        # Calculate error using Euclidean Distance formula
        euclidean_distance = sqrt((r_translation - best_r_translation)**2 + (c_translation - best_c_translation)**2)

        # if the error is greater than the threshold
        if euclidean_distance > threshold:
            # Update row and column translation total
            r_translation_total = r_translation
            c_translation_total = c_translation
            # Compare each translation, with every other translation
            for k in range(len(mappings)):
                # Except for itself..
                if k != i:
                    r_translation_new = translations[k][0]
                    c_translation_new = translations[k][1]
                    # Calculate updated error
                    euclidean_distance = sqrt((r_translation - r_translation_new) ** 2
                                              + (c_translation - c_translation_new) ** 2)
                    # If the error is less than the threshold
                    if euclidean_distance <= threshold:
                        # Update row and column translation total for that mapping and the number of agreements
                        r_translation_total += r_translation_new
                        c_translation_total += c_translation_new
                        strong_agreements += 1
                    # Otherwise update number of bad agreements
                    else:
                        weak_agreements += 1
                # Calculate best row and column offset
                best_row_translation = r_translation_total / strong_agreements
                best_column_translation = c_translation_total / strong_agreements

    print(f"Total number of matches:   {len(mappings)}")
    print(f"Strong matches:            {strong_agreements}")
    print(f"Weak matches:              {weak_agreements}")

    return best_row_translation, best_column_translation
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def compose_images(image1, image2, dr, dc):
    """ Make an empty white background depending on the on the values of dr and dc and
        paste the input images accordingly. """
    # Case: 1
    if dr > 0 and dc > 0:
        background = Image.new('RGB', (int(dc) + image2.width, int(dr) + image2.height), color='white')
        background.paste(image1, (0,0))
        background.paste(image2, (int(dc), int(dr)))
    # Case: 2
    elif dr < 0 and dc > 0:
        background = Image.new('RGB', (int(dc) + image2.width, int(abs(dr)) + image1.height), color='white')
        background.paste(image1, (0, int(abs(dr))))
        background.paste(image2, (int(dc), 0))
    # Case: 3
    elif dr > 0 and dc < 0:
        background = Image.new('RGB', (int(abs(dc)) + image1.width, int(dr) + image2.height), color='white')
        background.paste(image1, (int(abs(dc)), 0))
        background.paste(image2, (0, int(dr)))
    # Case: 4
    else:
        background = Image.new('RGB', (int(abs(dc)) + image1.width, int(abs(dr)) + image1.height), color='white')
        background.paste(image1, (int(abs(dc)), int(abs(dr))))
        background.paste(image2, (0,0))

    plt.imshow(background)
    plt.axis('off')
    plt.show()
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Open the images, convert to grayscale and normalize
    image_1 = np.array(Image.open("./Test Images/al1.png").convert('L'))/255.0
    image_2 = np.array(Image.open("./Test Images/al2.png").convert('L'))/255.0

    # 1. Form Harris responses and find Harris interest points
    harris_response_image_1 = get_harris_response(image_1, 1)
    harris_response_image_2 = get_harris_response(image_2, 1)

    interest_points_image_1 = get_harris_points(harris_response_image_1, 0.15, 10)
    interest_points_image_2 = get_harris_points(harris_response_image_2, 0.15, 10)

    plot_harris_points(image_1, interest_points_image_1)
    plot_harris_points(image_2, interest_points_image_2)

    #  2. Form normalized patch descriptor vectors
    descriptors_image_1 = get_descriptors(image_1, interest_points_image_1, 5)
    descriptors_image_2 = get_descriptors(image_2, interest_points_image_2, 5)

    # 3. Form response matrix
    coordinate_pairs = get_response_matrix(descriptors_image_1, descriptors_image_2, 0.95)
    plot_mapping(coordinate_pairs, image_1, image_2, interest_points_image_1, interest_points_image_2)

    # 4. Determine best translation using RANSAC
    r_offset, c_offset = get_best_translation(image_1, coordinate_pairs, interest_points_image_1,
                                              interest_points_image_2, 1.6)

    # 5. Compose the images
    test_image_1 = Image.open('./Test Images/al1.png')
    test_image_2 = Image.open('./Test Images/al2.png')
    compose_images(test_image_1, test_image_2, r_offset, c_offset)
# ----------------------------------------------------------------------------------------------------------------------

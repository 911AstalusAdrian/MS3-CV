'''
Write an application that uses a dataset of images (of at least) 20 images downloaded from the internet with various content (landscapes, objects, people, etc). 
The application should take an image as input argument and determine the set of similar images from the database by doing histogram comparisons. 
Suppose the set of all images is I={Ii, I2, …. In}. Extract an image Q from I and set it as the query image. DO the following steps:

1. Compute the histograms of all images in I and for the image Q.
2. Compare the histograms (using all metrics available in OpenCV for histogram comparison) of Q with Q and with all other image histograms from the set I.
3. Establish the maximum similarity as being the comp(Q,Q)  and normalize the results of other comparisons against this maximum similarity. 
   (Beware different metrics have different numerical values for the perfect similarity … (some zero, others large positive values, some 1, some large negative values, etc).
4. Use a second run with a color reduction mechanism where the histograms are computed on a smaller number of bins i.e. 64, 32 bins (each bin containing respectively 256/64 or 256/32 colors).

Compare the results and explain for each metric the results. What about after color reduction. Can we say that image retrieval based on graphical content be performed using histogram comparison?
You should print results in tabular numerical format where the comp(Q,Q) value is first displayed as reference values followed by all other similarities normalized.
'''



import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            img = cv2.imread(filepath)
            if img is not None:
                print(filepath)
                images.append((filename, img))
    return images


def plot_top_correlated_images(query_image, query_name, results, metric_name):
    # Sort results by similarity in descending order 
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    top_5 = sorted_results[:5]  # Select top 5 results

    # Set up the plot
    fig, axes = plt.subplots(1, 6, figsize=(15, 5))  # 1 row, 6 columns (query + 5)
    fig.suptitle(f"Top 5 Images for Metric: {metric_name}", fontsize=16)

    # Plot query image
    axes[0].imshow(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Query")
    axes[0].axis("off")

    # Plot top 5 correlated images
    for i, (name, _) in enumerate(top_5):
        img = cv2.imread(os.path.join(r'D:\Uni\MS3-CV\Labs\L4\images', name))
        axes[i + 1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i + 1].set_title(f"{name}")
        axes[i + 1].axis("off")

    # Show the plot
    plt.tight_layout()
    plt.show()

def compute_histogram(image, bins):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist

def compare_histograms(query_hist, histograms, comparison_methods):
    results = {method: [] for method in comparison_methods}
    for name, hist in histograms:
        for method in comparison_methods:
            sim = cv2.compareHist(query_hist, hist, method)
            results[method].append((name, sim))
    return results

def normalize_results(results, comparison_methods, query_query_comparisons):
    normalized_results = {}
    epsilon=1e-10
    for method in comparison_methods:
        if method == 0:
            q_q_value = query_query_comparisons[method]
            normalized_results[method] = [(name, (sim + 1) / 2 * q_q_value) for name, sim in results[method]]
        elif method == 2:
            q_q_value = query_query_comparisons[method]
            normalized_results[method] = [(name, sim / q_q_value) for name, sim in results[method]]
        else:
            normalized_results[method] = [(name, 1 /(1 + sim)) for name, sim in results[method]]
    return normalized_results

def print_results(normalized_results, comparison_methods, bins):
    print(f"\nResults with {bins} bins:")
    for method in comparison_methods:
        print(f"\nMetric: {method}")
        print(f"{'Image':<20}{'Normalized Similarity':<20}")
        for name, sim in normalized_results[method]:
            print(f"{name:<20}{sim:<20.6f}")

def main():
    dataset_folder = r'D:\Uni\MS3-CV\Labs\L4\images'

    images = load_images_from_folder(dataset_folder)

    if not images:
        print("No images found in the folder.")
        return

    index = 19
    # Query image: 
    query_name, query_image = images[index]

    # Parameters
    bins = 256
    comparison_methods = [
        cv2.HISTCMP_CORREL, 
        cv2.HISTCMP_CHISQR, 
        cv2.HISTCMP_INTERSECT, 
        cv2.HISTCMP_BHATTACHARYYA
    ]

    print(f"Query image: {query_name}")

    # Compute histograms
    histograms = [(name, compute_histogram(img, bins)) for name, img in images]
    query_hist = compute_histogram(query_image, bins)

    # Compare histograms
    results = compare_histograms(query_hist, histograms, comparison_methods)

    # Compute Q-Q values for normalization
    query_query_comparisons = {method: cv2.compareHist(query_hist, query_hist, method) for method in comparison_methods}

    # Normalize results
    normalized_results = normalize_results(results, comparison_methods, query_query_comparisons)

    # Print results
    print_results(normalized_results, comparison_methods, bins)

    for method in comparison_methods:
        metric_name = {
            cv2.HISTCMP_CORREL: "Correlation",
            cv2.HISTCMP_CHISQR: "Chi-Square",
            cv2.HISTCMP_INTERSECT: "Intersection",
            cv2.HISTCMP_BHATTACHARYYA: "Bhattacharyya"
        }[method]

        print(f"\nPlotting top 5 correlated images for metric: {metric_name} on {bins} bins")
        plot_top_correlated_images(query_image, query_name, normalized_results[method], metric_name)
        

if __name__ == "__main__":
    main()
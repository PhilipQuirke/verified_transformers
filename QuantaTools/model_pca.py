import torch
import numpy as np
import warnings
from QuantaTools.useful_node import NodeLocation
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler


# Calculate the percentage of the first Principal Component Analysis
# Returns 0 to 100
def pca_evr_0_percent(pca):
    return int(round(pca.explained_variance_ratio_[0] * 100, 0))


# Analyze the clusters of the PCA outputs looking for existance of 2 or 3 clusters
def analyze_pca_clusters(pca_outputs, true_labels, n_init=10):
    
    # Standardize the PCA outputs
    scaler = StandardScaler()
    pca_outputs_scaled = scaler.fit_transform(pca_outputs)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        # Try clustering with 2 and 3 clusters
        kmeans_2 = KMeans(n_clusters=2, n_init=n_init, random_state=42)
        kmeans_3 = KMeans(n_clusters=3, n_init=n_init, random_state=42)

        labels_2 = kmeans_2.fit_predict(pca_outputs_scaled)
        labels_3 = kmeans_3.fit_predict(pca_outputs_scaled)

        # Evaluate clustering quality
        silhouette_2 = silhouette_score(pca_outputs_scaled, labels_2)
        silhouette_3 = silhouette_score(pca_outputs_scaled, labels_3)

        calinski_2 = calinski_harabasz_score(pca_outputs_scaled, labels_2)
        calinski_3 = calinski_harabasz_score(pca_outputs_scaled, labels_3)

    # Compare clustering results with true labels
    def cluster_label_agreement(cluster_labels, true_labels):
        unique_clusters = np.unique(cluster_labels)
        agreement_scores = []

        for cluster in unique_clusters:
            cluster_mask = (cluster_labels == cluster)
            cluster_true_labels = true_labels[cluster_mask]
            most_common_label = np.argmax(np.bincount(cluster_true_labels))
            agreement_score = np.mean(cluster_true_labels == most_common_label)
            agreement_scores.append(agreement_score)

        return np.mean(agreement_scores)

    agreement_2 = cluster_label_agreement(labels_2, true_labels)
    agreement_3 = cluster_label_agreement(labels_3, true_labels)

    return {
        "silhouette_scores": {"2_clusters": silhouette_2 * 100, "3_clusters": silhouette_3 * 100},
        "calinski_harabasz_scores": {"2_clusters": calinski_2, "3_clusters": calinski_3},
        "label_agreement_scores": {"2_clusters": agreement_2 * 100, "3_clusters": agreement_3 * 100}
    }


# Calculate one Principal Component Analysis on test_inputs.
# Assume test_inputs is a list of questions of three types. 
def calc_pca_for_an(cfg, node_location : NodeLocation, test_inputs, title, error_message):
    assert node_location.is_head is True

    try:
        _, the_cache = cfg.main_model.run_with_cache(test_inputs)

        # Gather attention patterns for all the questions
        attention_outputs = []
        for i in range(len(test_inputs)):

          # Output of individual heads, without final bias
          attention_cache=the_cache["result", node_location.layer, "attn"] # Output of individual heads, without final bias
          attention_output=attention_cache[i]  # Shape [n_ctx, n_head, d_model]
          attention_outputs.append(attention_output[node_location.position, node_location.num, :])

        attn_outputs = torch.stack(attention_outputs, dim=0).cpu()

        pca = PCA(n_components=6)
        pca.fit(attn_outputs)
        pca_attn_outputs = pca.transform(attn_outputs)
       
        # Create true_labels assuming input is 3 types of questions
        n_questions = len(test_inputs) // 3
        assert n_questions > 5
        assert n_questions * 3 == len(test_inputs)
        true_labels = np.array([0]*n_questions + [1]*n_questions + [2]*n_questions)  # 0 for Type A, 1 for Type B, 2 for Type C

        # Analyze output testing for existance of 2 or 3 clusters
        cluster_results = analyze_pca_clusters(pca_attn_outputs, true_labels)

        evr_perc = pca_evr_0_percent(pca)
        silhouette_scores = cluster_results['silhouette_scores']
        calinski_harabasz_scores = cluster_results['calinski_harabasz_scores']
        label_agreement_scores = cluster_results['label_agreement_scores']
        silhouette_score = max( silhouette_scores['2_clusters'], silhouette_scores['3_clusters'] ) 
        calinski_harabasz_score = max( calinski_harabasz_scores['2_clusters'], calinski_harabasz_scores['3_clusters'])
        label_agreement_score = max( label_agreement_scores['2_clusters'], label_agreement_scores['3_clusters']) 

        # Full_title is "P10L0H1 A3 78/43/62/42" = NodeLocation AnswerDigit EVR[0]/MaxSilhouetteScore/MaxCalinskiHarabaszScore/MaxLabelAgreementScore
        full_title = format( f"{title} {evr_perc}/{silhouette_score:.0f}/{calinski_harabasz_score:.0f}/{label_agreement_score:.0f}" )             

        return pca, pca_attn_outputs, full_title, cluster_results
    
    except Exception as e:
        #print(error_message, e)
        return None, None, "", None
    

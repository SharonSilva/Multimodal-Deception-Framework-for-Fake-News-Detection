# """
# run_clustering_pipeline.py
# ===========================
# Simple script to run the complete clustering pipeline.
# """

# import torch
# import pandas as pd
# from deception_clustering import IntelligentDeceptionPatternClustering

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print("="*70)
# print("RUNNING INTELLIGENT DECEPTION PATTERN CLUSTERING")
# print("="*70)

# # ============================================================================
# # STEP 1: Load prepared data
# # ============================================================================

# print("\n[1/3] Loading prepared data...")
# data = torch.load("prepared_clustering_data.pt")

# print(f"✅ Data loaded:")
# print(f"   Posts: {data['metadata']['n_posts']}")
# print(f"   Users: {data['metadata']['n_users']}")
# print(f"   z_out: {data['z_out'].shape}")
# print(f"   v_mismatch: {data['v_mismatch'].shape}")

# # ============================================================================
# # STEP 2: Initialize and run pipeline
# # ============================================================================

# print("\n[2/3] Initializing clustering pipeline...")

# pipeline = IntelligentDeceptionPatternClustering(
#     z_out_dim=data['z_out'].shape[1],
#     mismatch_dim=data['v_mismatch'].shape[1],
#     metadata_dim=0,
#     clustering_method='kmeans',
#     n_clusters=8,
#     tau=3600.0,  # 1 hour time decay
#     device=device
# )

# print("✅ Pipeline initialized")

# # Run clustering
# print("\n[3/3] Running clustering pipeline...")
# pipeline.fit(
#     z_out=data['z_out'],
#     v_mismatch=data['v_mismatch'],
#     user_ids=data['user_ids'],
#     timestamps=data['timestamps']
# )

# # ============================================================================
# # STEP 3: Analyze results
# # ============================================================================

# print("\n" + "="*70)
# print("ANALYSIS RESULTS")
# print("="*70)

# # Get high-risk users
# high_risk = pipeline.get_high_risk_users(top_k=20, threshold=0.5)

# print(f"\n🚨 HIGH-RISK USERS (Top 20)")
# print("="*70)

# for i, (user_id, scores) in enumerate(high_risk, 1):
#     print(f"\n{i}. User: {user_id}")
#     print(f"   📊 Campaign Score: {scores['campaign_score']:.3f}")
#     print(f"   └─ Cluster Coherence: {scores['cluster_coherence']:.3f}")
#     print(f"   └─ Temporal Synchrony: {scores['temporal_synchrony']:.3f}")
#     print(f"   └─ Motif Frequency: {scores['motif_frequency']:.3f}")
#     print(f"   └─ Mismatch Magnitude: {scores['mismatch_magnitude']:.3f}")
#     print(f"   📝 Total Posts: {scores['n_posts']}")

# # Generate report
# report = pipeline.generate_report()

# print("\n" + "="*70)
# print("SYSTEM REPORT")
# print("="*70)

# print(f"\n📊 Clustering:")
# print(f"   Strategies Detected: {report['clustering']['n_clusters']}")
# print(f"   Silhouette Score: {report['clustering']['silhouette_score']:.3f}")

# print(f"\n👥 Users:")
# print(f"   Total: {report['n_users']}")
# print(f"   High-Risk: {report['score_stats']['high_risk_count']}")

# print(f"\n🌐 Communities:")
# print(f"   Detected: {report['n_communities']}")

# print(f"\n🔍 Motifs:")
# print(f"   Found: {report['n_motifs']}")
# print(f"   Top 5:")
# for motif, stats in report['top_motifs']:
#     print(f"   {motif} → lift={stats['lift']:.2f}")

# print(f"\n📈 Scores:")
# print(f"   Mean: {report['score_stats']['mean']:.3f}")
# print(f"   Max: {report['score_stats']['max']:.3f}")

# # Interpret clusters
# print("\n" + "="*70)
# print("INTERPRETING MANIPULATION STRATEGIES")
# print("="*70)

# df = pd.read_pickle("Dataset/twitter/df_with_all_features.pkl")
# interpretations = pipeline.interpret_clusters(
#     df=df,
#     z_out=data['z_out'],
#     v_mismatch=data['v_mismatch'],
#     sample_size=3
# )

# # Save results
# print("\n💾 Saving results...")
# pipeline.save_results(output_dir='clustering_results')

# print("\n" + "="*70)
# print("✅ CLUSTERING COMPLETE!")
# print("="*70)

# print("\n📁 Results saved to clustering_results/:")
# print("   - campaign_scores.csv")
# print("   - cluster_labels.pkl")
# print("   - motifs.pkl")
# print("   - temporal_graph.gpickle")
# print("   - cluster_interpretations.pkl")
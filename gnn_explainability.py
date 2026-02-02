# """
# ===========================================================
# EXPLAINABILITY ENGINE FOR TEMPORAL HETEROGENEOUS GNN
# ===========================================================

# This module extracts latent explainability signals from:
# - Heterogeneous relations
# - Temporal evolution
# - Coordination patterns
# - Community & deception clusters

# WITHOUT modifying model training.

# Outputs:
# - Structured JSON explanations
# - Human-readable natural language explanations

# Author: (You)
# ===========================================================
# """

# import torch
# import numpy as np
# from collections import defaultdict


# # =========================================================
# # 1️⃣ RELATION-AWARE EXPLAINER
# # =========================================================
# class RelationExplainer:
#     def __init__(self, relation_weights):
#         """
#         relation_weights: dict {relation_type: Tensor}
#         """
#         self.relation_weights = relation_weights

#     def explain(self, messages_per_relation):
#         """
#         messages_per_relation: dict {relation_type: Tensor [num_msgs, dim]}
#         """
#         contributions = {}

#         for relation, messages in messages_per_relation.items():
#             if messages.numel() == 0:
#                 continue

#             message_strength = torch.norm(messages, dim=1).mean().item()
#             relation_weight = torch.norm(self.relation_weights[relation]).item()

#             contributions[relation] = message_strength * relation_weight

#         total = sum(contributions.values()) + 1e-9

#         explanation = [
#             {
#                 "relation": rel,
#                 "importance": round(score / total, 3)
#             }
#             for rel, score in contributions.items()
#         ]

#         return sorted(explanation, key=lambda x: x["importance"], reverse=True)


# # =========================================================
# # 2️⃣ TEMPORAL EXPLAINER (WHY NOW?)
# # =========================================================
# class TemporalExplainer:
#     def explain(self, temporal_embeddings, time_deltas):
#         """
#         temporal_embeddings: Tensor [T, D]
#         time_deltas: list or array [T]
#         """
#         embedding_norms = torch.norm(temporal_embeddings, dim=1)
#         deltas = torch.abs(embedding_norms[1:] - embedding_norms[:-1])

#         avg_gap = float(np.mean(time_deltas))
#         recent_gap = float(time_deltas[-1])

#         temporal_strength = "high" if deltas.mean() > deltas.std() else "moderate"

#         return {
#             "recent_activity_boost": recent_gap < avg_gap * 0.5,
#             "avg_time_gap_seconds": round(avg_gap, 2),
#             "temporal_influence_strength": temporal_strength
#         }


# # =========================================================
# # 3️⃣ COORDINATION & CAMPAIGN EXPLAINER
# # =========================================================
# class CoordinationExplainer:
#     def explain(self, edge_scores, threshold=0.5):
#         """
#         edge_scores: Tensor [num_edges]
#         """
#         suspicious_edges = (edge_scores > threshold).float()
#         ratio = suspicious_edges.mean().item()

#         if ratio >= 0.6:
#             level = "high"
#         elif ratio >= 0.3:
#             level = "moderate"
#         else:
#             level = "low"

#         return {
#             "suspicious_edges_ratio": round(ratio, 3),
#             "coordination_level": level,
#             "dominant_pattern": (
#                 "user-to-user reinforcement"
#                 if level == "high"
#                 else "weak or isolated interactions"
#             )
#         }


# # =========================================================
# # 4️⃣ COMMUNITY & CLUSTER EXPLAINER
# # =========================================================
# class CommunityExplainer:
#     def explain(self, community_risk_score, cluster_deception_score):
#         community_risk_score = float(community_risk_score)
#         cluster_deception_score = float(cluster_deception_score)

#         explanation = (
#             "Community shows sustained deceptive behavior"
#             if community_risk_score > 0.7
#             else "Community behavior appears largely normal"
#         )

#         return {
#             "community_risk": round(community_risk_score, 3),
#             "cluster_deception_score": round(cluster_deception_score, 3),
#             "explanation": explanation
#         }


# # =========================================================
# # 5️⃣ NATURAL LANGUAGE EXPLAINER
# # =========================================================
# class NaturalLanguageExplainer:
#     def explain(self, relation_exp, temporal_exp, coord_exp, community_exp):
#         reasons = []

#         if relation_exp:
#             top_rel = relation_exp[0]
#             reasons.append(
#                 f"strong interactions via {top_rel['relation']}"
#             )

#         if temporal_exp["recent_activity_boost"]:
#             reasons.append("a sudden burst of recent activity")

#         if coord_exp["coordination_level"] == "high":
#             reasons.append("coordinated behavior among multiple users")

#         if community_exp["community_risk"] > 0.7:
#             reasons.append("a high-risk community context")

#         if not reasons:
#             return "This post was flagged due to subtle anomalous patterns detected by the model."

#         return (
#             "This post was flagged as deceptive because it exhibits "
#             + ", ".join(reasons)
#             + "."
#         )


# # =========================================================
# # 6️⃣ MASTER EXPLAINABILITY ENGINE
# # =========================================================
# class ExplainabilityEngine:
#     def __init__(self, relation_weights):
#         self.relation_explainer = RelationExplainer(relation_weights)
#         self.temporal_explainer = TemporalExplainer()
#         self.coordination_explainer = CoordinationExplainer()
#         self.community_explainer = CommunityExplainer()
#         self.nl_explainer = NaturalLanguageExplainer()

#     def generate_explanation(self, inputs):
#         """
#         inputs dict must contain:
#         {
#             "messages_per_relation": dict,
#             "temporal_embeddings": Tensor,
#             "time_deltas": list,
#             "edge_scores": Tensor,
#             "community_risk": float,
#             "cluster_score": float
#         }
#         """

#         relation_exp = self.relation_explainer.explain(
#             inputs["messages_per_relation"]
#         )

#         temporal_exp = self.temporal_explainer.explain(
#             inputs["temporal_embeddings"],
#             inputs["time_deltas"]
#         )

#         coord_exp = self.coordination_explainer.explain(
#             inputs["edge_scores"]
#         )

#         community_exp = self.community_explainer.explain(
#             inputs["community_risk"],
#             inputs["cluster_score"]
#         )

#         nl_exp = self.nl_explainer.explain(
#             relation_exp, temporal_exp, coord_exp, community_exp
#         )

#         return {
#             "relation_aware_explanation": relation_exp,
#             "temporal_explanation": temporal_exp,
#             "coordination_explanation": coord_exp,
#             "community_explanation": community_exp,
#             "final_natural_language_explanation": nl_exp
#         }


# # =========================================================
# # ✅ END OF FILE
# # =========================================================
# # =========================================================
# # 🔽 EXAMPLE USAGE & OUTPUT GENERATION
# # =========================================================
# if __name__ == "__main__":

#     print("=" * 80)
#     print("RUNNING EXPLAINABILITY ENGINE")
#     print("=" * 80)

#     # -----------------------------------------------------
#     # Simulated model outputs (normally from your GNN)
#     # -----------------------------------------------------

#     # Relation weights (from HeteroMessagePassing)
#     relation_weights = {
#         "user_interacts_with_user": torch.randn(256, 256),
#         "post_flagged_in_deception_cluster": torch.randn(256, 256),
#         "deception_cluster_colludes_with_deception_cluster": torch.randn(256, 256)
#     }

#     # Messages aggregated per relation
#     messages_per_relation = {
#         "user_interacts_with_user": torch.randn(40, 256),
#         "post_flagged_in_deception_cluster": torch.randn(25, 256),
#         "deception_cluster_colludes_with_deception_cluster": torch.randn(10, 256)
#     }

#     # Temporal embeddings (history of node over time)
#     temporal_embeddings = torch.randn(6, 256)
#     time_deltas = [3600, 1800, 900, 600, 300, 120]  # seconds

#     # Edge-level coordination scores
#     edge_scores = torch.rand(50)  # from edge_classifiers

#     # Community & cluster scores
#     community_risk = 0.82
#     cluster_score = 0.91

#     # -----------------------------------------------------
#     # Run explainability
#     # -----------------------------------------------------
#     explainer = ExplainabilityEngine(relation_weights)

#     explanation = explainer.generate_explanation({
#         "messages_per_relation": messages_per_relation,
#         "temporal_embeddings": temporal_embeddings,
#         "time_deltas": time_deltas,
#         "edge_scores": edge_scores,
#         "community_risk": community_risk,
#         "cluster_score": cluster_score
#     })

#     # -----------------------------------------------------
#     # OUTPUT
#     # -----------------------------------------------------
#     print("\n📊 STRUCTURED EXPLANATION (JSON-like)")
#     print("-" * 80)
#     for key, value in explanation.items():
#         print(f"\n{key.upper()}:")
#         print(value)

#     print("\n🧠 HUMAN-READABLE EXPLANATION")
#     print("-" * 80)
#     print(explanation["final_natural_language_explanation"])

#     print("\n✅ EXPLAINABILITY COMPLETED")
#     print("=" * 80)
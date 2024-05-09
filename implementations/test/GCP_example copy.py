from quBLP.problemtemplate import GraphColoringProblem as GCP
gcp = GCP(3,[[0, 1], [1, 2]], False)
# gcp = GCP(4,[[0, 1], [1, 2], [2, 3], [3, 0]], False)
gcp.optimize()
print(gcp.solution)
from quBLP.problemtemplate import GraphColoringProblem as GCP
gcp = GCP(3,[[0, 1], [1, 2]], False)
gcp.optimize(30,0.01,2)
print(gcp.solution)
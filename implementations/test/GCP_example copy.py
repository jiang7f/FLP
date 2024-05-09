from quBLP.problemtemplate import GraphColoringProblem as GCP
gcp = GCP(3,[[0, 1], [1, 2]], False)
# gcp = GCP(4,[[0, 1], [1, 2], [2, 3], [3, 0]], False)
# 这个层数和迭代次数要稍微多一点.
gcp.optimize(30, 0.1, 5)
# print(gcp.solution)
import math

hes_scores = []
results = []
file_name = "/gs/fs/tga-aklab/matsumoto/Main/log/24-10-12 20-20-41/eval5/description.txt" # Novelty
file_name = "/gs/fs/tga-aklab/matsumoto/Main/log/24-10-12 06-43-01/eval5/description.txt" # Coverage

with open(file_name, "r") as file:
    for i, line in enumerate(file):
        if (i + 1) % 5 == 0:
            score = float(line.strip())
            if score > 5.0:
                score = 5.0
            elif score < 1.0:
                score = 1.0
            hes_scores.append(score)
            score = (2 / (1+math.exp(-(score-5))))
            results.append(score)

hes_avg = sum(hes_scores) / len(hes_scores)
results_avg = sum(results) / len(results)

print(file_name)
print(f"HES SCORE: {hes_avg}")
print(f"change: {results_avg}")
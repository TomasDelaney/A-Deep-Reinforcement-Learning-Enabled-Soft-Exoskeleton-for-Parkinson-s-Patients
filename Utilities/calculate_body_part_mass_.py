
'''
This script calculates the forearm and upper arm masses based on:
    STANLEY PLAGENHOEF: Anatomical Data for Analyzing Human Motion research paper
'''


def calculate_body_part_mass(mass):
    # these percentages are present in MALES
    hum_weight = mass * 3.25 / 100
    forearm_weight = mass * 1.87 / 100
    hand_weight = mass * 0.65 / 100

    return hum_weight, forearm_weight, hand_weight


if __name__ == "__main__":
    mass = 81.5     # mass is in kg
    print(calculate_body_part_mass(mass))

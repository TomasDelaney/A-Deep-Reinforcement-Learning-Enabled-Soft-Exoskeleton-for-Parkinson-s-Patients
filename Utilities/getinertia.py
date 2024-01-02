'''
This script refers to the study called: Models and Modelling of Dynamic Moments of Inertia of Human Body
- here we estimate the moments of inertia of the upper and lower arm by calculating the inertia's of certain cylinders
- the used parameters of the cylinder:
    m: mass of the cylinder (kg)
    r: radius of the cylinder (m)
    h: height of the cylinder (m)
the reference for the formula: Machinery’s Handbook 29th Edition
'''


def calculate_moments_of_inertia(mass, radius, height):
    # calculates the inertia's at the central diameter
    i_x = (mass/12) * (3 * radius ** 2 + height ** 2)
    i_y = (mass/12) * (3 * radius ** 2 + height ** 2)
    i_z = (mass * radius ** 2) / 2

    inertias = [i_x, i_y, i_z]

    return inertias


def calculate_moments_of_inertia2(mass, radius, height):
    # calculates the inertia's at the end of the diameter
    i_x = mass * ((radius ** 2)/4 + (height ** 2) / 3)
    i_y = mass * ((radius ** 2)/4 + (height ** 2) / 3)
    i_z = (mass * radius ** 2) / 2

    inertias = [i_x, i_y, i_z]

    return inertias


def calculate_moments_of_inertia_hand(hand_mass, fist_radius):
    # calculates the hand inertia where Ix = Iy = Iz, hand is approximated as a solid sphere
    inertia = (2 / 5) * hand_mass * fist_radius**2
    return inertia

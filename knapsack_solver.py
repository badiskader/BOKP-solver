import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_option_menu import option_menu
import time

def knapsack_solver(poids_maximum, poids_object, z1_vect, z2_vect, epsilon):
    start = time.time() # pour calculer le temps de calculation.
    decision = cp.Variable(len(poids_object), boolean=True) # initisaliser les variables (dans ce cas elles sont des variable des decision)
    contrainte_poid = poids_object @ decision <= poids_maximum # contrainte de poid (la somme des poids d'objets doit etre inferieur ou egale au poid max)
    Z1 = z1_vect @ decision # initialiser une variable Z1 qui contient la somme des valeur de la premier fonction objectif des objets de la solution
    p1 = cp.Problem(cp.Maximize(Z1), [contrainte_poid]) # initialiser le problem de maximisation de Z1 avec la contrainte de poid
    p1.solve(solver=cp.GLPK_MI) # resoudre le probleme (P1)

    Z2 = z2_vect @ decision # initialiser une variable Z2 qui contient la somme des valeur de la deuxieme fonction objectif des objets de la solution trouver
    p2 = cp.Problem(cp.Maximize(Z2), [contrainte_poid, Z1 == Z1.value]) # resoudre le probleme de maximisation de Z2 avec la contrainte depoid
                                                                        # et la valuer de z1 est la valeur trouvée dans la solution précédente
    p2.solve(solver=cp.GLPK_MI) # resoudre le probleme (P2)

    solutions_pareto = [decision.value]
    j = 1
    valeurs_solution_Z1 = [Z1.value]
    valeurs_solution_Z2 = [Z2.value]
    solutions_Z = [(Z1.value, Z2.value)]

    while True:
        p_epsilon = cp.Problem(cp.Maximize(Z1), [contrainte_poid, Z2 >= valeurs_solution_Z2[j - 1] + epsilon])
        p_epsilon.solve(solver=cp.GLPK_MI)

        if (Z1.value, Z2.value) == (None, None):
            break
        else:
            solutions_pareto.append(decision.value)
            valeurs_solution_Z1.append(Z1.value)
            valeurs_solution_Z2.append(Z2.value)
            solutions_Z.append((Z1.value, Z2.value))
            j += 1
    end = time.time()

    st.write("### L'ensemble des solutions efficaces:")
    indexes = []
    for i in range(len(solutions_pareto)):
        indexes.append(f'X({i+1})')
    df = pd.DataFrame({'': indexes,'Solution:': solutions_pareto, 'Z1': valeurs_solution_Z1, 'Z2': valeurs_solution_Z2})
    table = st.expander("Solutions")
    with table:
        st.table(df.set_index(df.columns[0]))

    st.write("### Front de pareto:")
    plt.figure(figsize=(8, 6))
    plt.plot(valeurs_solution_Z1, valeurs_solution_Z2, marker='o', linestyle='-')
    plt.xlabel('Z1')
    plt.ylabel('Z2')
    plt.title('Front de pareto')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    st.write(f'### Temps de calcul: {round((end - start), 4)} seconds')


def main():
    selected = option_menu(
        menu_title= "",
        options= ["Manual", "Random"],
        default_index=0,
        orientation='horizontal'
    )

    if selected == "Manual":
        text = "Developer par: Boussaa Abderrahmane et Badis Abdelkader Amine"
        bold_text = f"<p style='font-weight: bold;'>{text}</p>"

        st.write(bold_text, unsafe_allow_html=True)
        st.title('Bi-objective Knapsack Problem Solver with the epsilon constraint method')
        poids_maximum = st.number_input('Poids maximum du sac:')
        n = st.number_input("Nombre d'objets:", step=1)
        epsilon = st.number_input("Valeur d'epsilon:")

        poids_object = np.array([])
        z1_vect = np.array([]) # initialiser un vecteur avec les valuers de la premier fonction objectif de touts les objets
        z2_vect = np.array([])

        solve = st.button('Résoudre')
        st.write()
        st.write("### L'ensemble des objets:")
        with st.expander('Objets:'):
            for i in range(int(n)):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write(f'## Objet {i + 1}')
                with col2:
                    poid = st.number_input(f"Poids de l'objet:", key=f"poids_{i+1}")
                with col3:
                    valeur1 = st.number_input(f"Valeur 1 de l'objet:", key=f"z1_{i+1}")
                with col4:
                    valeur2 = st.number_input(f"Valeur 2 de l'objet:", key=f"z2_{i+1}")

                poids_object = np.append(poids_object, poid)
                z1_vect = np.append(z1_vect, valeur1)
                z2_vect = np.append(z2_vect, valeur2)
        if solve:
            knapsack_solver(poids_maximum, poids_object, z1_vect, z2_vect, epsilon)
    if selected == "Random" :
        text = "Developer par: Boussaa Abderrahmane et Badis Abdelkader Amine"
        bold_text = f"<p style='font-weight: bold;'>{text}</p>"
        st.write(bold_text, unsafe_allow_html=True)
        st.title('Bi-objective Knapsack Problem Solver with the epsilon constraint method')
        poids_maximum = st.number_input('Poids maximum du sac:')
        n = st.number_input("Nombre d'objets:", step=1)
        epsilon = st.number_input("Valeur d'epsilon:")

        poids_object = np.array([])
        z1_vect = np.array([])
        z2_vect = np.array([])

        random = st.button('Randomize')
        if random:
            randomized = True
            st.write("### L'ensemble des objets:")
            with st.expander('Objets:'):
                for i in range(int(n)):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.write(f'### Objet {i + 1}')
                    with col2:
                        poid = st.number_input(f"Poids de l'objet:", key=f"Rpoids_{i+1}", value= np.random.randint(0,poids_maximum) + np.random.random())
                    with col3:
                        valeur1 = st.number_input(f"Valeur 1 de l'objet:", key=f"Rz1_{i+1}",value= abs(np.random.randint(0, 100) + np.random.random()))
                    with col4:
                        valeur2 = st.number_input(f"Valeur 2 de l'objet:", key=f"Rz2_{i+1}",value= abs(np.random.randint(0, 100) + np.random.random()))

                    poids_object = np.append(poids_object, poid)
                    z1_vect = np.append(z1_vect, valeur1)
                    z2_vect = np.append(z2_vect, valeur2)
            knapsack_solver(poids_maximum, poids_object, z1_vect, z2_vect, epsilon)


if __name__ == '__main__':
    main()

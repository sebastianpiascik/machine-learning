from pyeasyga import pyeasyga

data = [{'id': 1, 'przedmiot':  'zegar', 'value': 100, 'weight': 7},
        {'id': 2, 'przedmiot':  'obraz-pejzaz', 'value': 300, 'weight': 7},
        {'id': 3, 'przedmiot':  'obraz-portret', 'value': 200, 'weight': 6},
        {'id': 4, 'przedmiot':  'radio', 'value': 40, 'weight': 2},
        {'id': 5, 'przedmiot':  'laptop', 'value': 500, 'weight': 5},
        {'id': 6, 'przedmiot':  'lampka nocna', 'value': 70, 'weight': 6},
        {'id': 7, 'przedmiot':  'srebne sztućce', 'value': 100, 'weight': 1},
        {'id': 8, 'przedmiot':  'porcelana', 'value': 250, 'weight': 3},
        {'id': 9, 'przedmiot':  'figura z brazu', 'value': 300, 'weight': 10},
        {'id': 10, 'przedmiot':  'skorzana torebka', 'value': 280, 'weight': 3},
        {'id': 11, 'przedmiot':  'odkurzacz', 'value': 300, 'weight': 15}]

ga = pyeasyga.GeneticAlgorithm(data,
                               population_size=200,
                               generations=2,
                               mutation_probability=0.05,
                               elitism=True)


def fitness (individual, data):
    values, weights = 0, 0
    for selected, box in zip(individual, data):
        if selected:
            values += box.get('value')
            weights += box.get('weight')
    if weights > 25:
        values = 0
    return values

ga.fitness_function = fitness
ga.run()
print (ga.best_individual())

for individual in ga.last_generation():
     print (ga.last_generation()[0])


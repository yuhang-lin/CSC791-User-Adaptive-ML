from genetic import Genetic

def repeat(length, numRound=20):
    genetic = Genetic(4)
    for i in range(numRound):
        genetic.main(length)

if __name__ == "__main__":
    num_round = 5
    for i in range(4, 9):
        repeat(i, num_round)

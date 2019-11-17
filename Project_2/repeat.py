from genetic import Genetic

def repeat(length, numRound=20):
    genetic = Genetic(4)
    for i in range(numRound):
        genetic.main(length)

if __name__ == "__main__":
    repeat(4, 3)
    repeat(5, 3)
    repeat(6, 3)

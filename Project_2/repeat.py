from genetic import Genetic

def repeat(length, numRound=20):
    genetic = Genetic(2)
    for i in range(numRound):
        genetic.main(length)

if __name__ == "__main__":
    repeat(8, 20)
    repeat(7, 20)
    repeat(6, 20)
    repeat(5, 20)

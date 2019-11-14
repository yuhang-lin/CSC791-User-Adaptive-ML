import genetic

def repeat(length, numRound=20):
    for i in range(numRound):
        genetic.main(length)

if __name__ == "__main__":
    repeat(8, 20)

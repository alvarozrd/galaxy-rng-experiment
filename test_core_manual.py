from src.rng.core import PCG64CPALite1


gerador = PCG64CPALite1(seed=123456789)

print("Cinco inteiros de 64 bits:")
for _ in range(5):
    print(gerador.next_u64())

print("\nCinco valores em [0, 1):")
for _ in range(5):
    print(gerador.random_float())
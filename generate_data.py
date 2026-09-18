import os
os.makedirs('data', exist_ok=True)
samples = [
    "I seen him go to the bank yesterday. He deposit all his money their. However, it was closed. Could you check if its open today?",
    "Me and him went to the park. The dog barked at the tree. John bought a wooden bat because he likes baseball. Then we went home.",
    "The manager say that the file is missing. Therefore, he need a new copy. Could you send the file to me?",
    "Its a beautiful day to go to the beach. The light is shining. She drove her car fast. But she got a ticket.",
    "Their going to the movies tonight. I wants to go to. Can you give me a ride?",
    "The student ate the apple. He like it very much. Furthermore, he ate a sandwich.",
    "Mary read the book. She thought it was good. Although, it was too long. Could you recommend another one?",
    "He send an email to the boss. The boss is angry because of the mistake. As a result, he fired him.",
    "The telescope is broken. The man saw the dog with it. Thus, he could not see the stars.",
    "I am going to the store to by some milk. My friend want some to. In addition, we need bread."
]
for i, s in enumerate(samples):
    with open(f'data/sample_{i+1}.txt', 'w', encoding='utf-8') as f:
        f.write(s)

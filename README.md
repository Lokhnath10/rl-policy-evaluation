# POLICY EVALUATION

## AIM
To simulate the Frozen-lake MDP and compare different policy functions.

## PROBLEM STATEMENT
The problem involves simulating a Frozen-lake MDP and defining various policy functions for it, these policy functions are later evaluated by a policy_evaluation() function which compares the value function of the policies passed as parameter. This is an experiment in reinforcement learning where you test different policies in FrozenLake, both by simulation (probability of reaching the goal) and by formal policy evaluation (computing expected long-term rewards).

## POLICY EVALUATION FUNCTION

<img width="685" height="130" alt="image" src="https://github.com/user-attachments/assets/834db01d-47b9-40d8-895e-5b7fc488ed1d" />

```python
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_v = np.zeros(len(P))  
    while True:
        v = np.zeros(len(P))  
        for s in range(len(P)):
            a = pi(s)  
            for prob, next_s, reward, done in P[s][a]:
                v[s] += prob * (reward + gamma * prev_v[next_s] * (not done))
        if np.max(np.abs(prev_v - v)) < theta:
            return v

        prev_v = v.copy()
    return v
```

## OUTPUT:
<img width="1446" height="616" alt="image" src="https://github.com/user-attachments/assets/c905cb6a-3b9a-4eef-97b1-602504d10fa9" />
<img width="1398" height="735" alt="image" src="https://github.com/user-attachments/assets/855fcfa4-8823-477b-bd00-098c67c9ecb4" />
<img width="1390" height="161" alt="image" src="https://github.com/user-attachments/assets/9f9afc17-72f8-496c-9918-17df441fc898" />
<img width="1025" height="498" alt="image" src="https://github.com/user-attachments/assets/30b50766-bf8b-4635-9c74-74d2e4a2d821" />
<img width="741" height="480" alt="image" src="https://github.com/user-attachments/assets/6f59df50-710e-48b2-9a1f-bc6a5de42674" />

## RESULT:
Thus we have successfully evaluated two different policies for a given env and compared their values functions.

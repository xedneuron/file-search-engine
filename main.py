import torch
import torch.nn.functional as F

# 1. Smaller initialization (multiply by 0.1) to prevent saturation
brain = (torch.randn(20, 20) * 0.1).detach().requires_grad_(True)

def vectorize(problem_str):
    vec = torch.zeros(20)
    # Mapping: 0-9 = 0-9, '+' = 10, '-' = 11
    tokens = [int(c) if c.isdigit() else (10 if c=='+' else 11) for c in problem_str.replace(" ", "")]
    for i, t in enumerate(tokens):
        vec[i] = t # Let's keep these as whole numbers for now
    return vec

def run_model(problem_vec):
    state = problem_vec
    
    for _ in range(3):
        # Thinker vs Adversary zones
        thinker_output  = state @ brain[:, 0:10]
        adversary_doubt = state @ brain[:, 10:20]
        
        # Parallel logic: Thinker minus a tiny bit of Adversary
        combined = thinker_output - (0.05 * adversary_doubt)
        
        # Navigation: Pass it back into the state
        # Using ReLU instead of tanh prevents the "Stuck at 10" problem
        state = torch.cat([combined, combined], dim=0)
        state = F.relu(state) 
        
    # We take the first value as our answer
    return state[0]

# 2. Training Loop
optimizer = torch.optim.Adam([brain], lr=0.005)
problems = ['1+1', '1+7', '2-6', '5+7', '9-7', '0+8', '4-1', '7+3', '9-5', '4+6', '9-0', '9+7', '0-1', '2+8', '5-6', '6+0', '7-8', '3+1', '8-4', '8+6', '8-3', '3+4', '2-1', '3+7', '0-5', '2+8', '0-9', '8+5', '2-8', '8+4', '8-9', '4+6', '5-4', '6+8', '6-4', '8+0', '3-9', '0+2', '0-1', '9+7', '7-9', '8+1', '6-8', '2+4', '1-6', '6+7', '2-0', '1+9', '9-5', '1+0', '4-7', '1+9', '9-9', '1+9', '0-4', '4+1', '3-9', '5+4', '4-8', '1+9', '2-3', '5+1', '5-3', '9+2', '2-6', '7+8', '8-8', '1+2', '1-6', '6+2', '5-2', '6+9', '2-5', '7+8', '6-4', '2+2', '8-5', '2+2', '3-6', '9+4', '1-6', '1+5', '0-5', '6+8', '8-6', '3+1', '0-3', '6+3', '9-3', '9+8', '4-2', '4+4', '0-3', '4+9', '5-2', '8+3', '2-5', '8+4', '5-8', '7+6', '4-6', '7+7', '0-1', '5+0', '2-2', '7+7', '8-3', '0+1', '7-3', '8+8', '4-3', '2+3', '7-4', '6+2', '1-7', '5+4', '0-6', '1+2', '4-0', '5+3', '4-9', '8+3', '5-4', '6+2', '0-7', '9+2', '2-6', '3+9', '3-1', '6+4', '2-4', '0+0', '2-4', '1+1', '5-9', '3+6', '1-1', '4+5', '2-6', '2+8', '6-5', '1+8', '8-5', '9+0', '2-1', '6+2', '9-6', '9+0', '0-0', '8+5', '7-1', '4+1', '9-7', '2+8', '3-8', '0+0', '5-7', '1+3', '8-0', '9+1', '0-2', '7+5', '2-2', '1+7', '1-8', '0+4', '7-3', '6+0', '4-8', '4+7', '1-3', '2+6', '9-7', '6+0', '3-1', '3+7', '1-2', '5+8', '1-8', '5+6', '5-7', '9+3', '2-0', '9+9', '5-0', '3+2', '1-4', '9+1', '5-0', '6+7', '6-1', '3+4', '1-6', '5+3', '7-8', '6+5', '3-6', '5+6', '8-5', '5+6', '3-4']

print("--- Starting Training ---")
for epoch in range(1001):
    total_loss = 0
    for prob in problems:
        target = float(eval(prob))
        prediction = run_model(vectorize(prob))
        
        # Mean Squared Error
        loss = F.mse_loss(prediction, torch.tensor(target))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Error is {total_loss:.4f}")

# 3. Test
test_prob = "4+2"
final_val = run_model(vectorize(test_prob))
print(f"\nTest Problem: {test_prob}")
print(f"Model Prediction: {final_val.item():.2f}")
print(f"Verified via eval(): {eval(test_prob)}")

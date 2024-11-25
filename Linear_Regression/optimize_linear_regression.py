import datetime
import json
import numpy as np
import os
import sys
from absl import app
from absl import flags
import google.generativeai as genai
import re
import time  
from dotenv import load_dotenv
load_dotenv()

def main(_):
    _GEMINI_API_KEY = API_KEY = ''
    genai.configure(api_key=_GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-pro")
    last_api_call_time = None
    
    # Initial parameters
    num_points = 50
    w_true = 20
    b_true = 6
    max_num_steps = 20
    num_reps = 1

    datetime_str = datetime.datetime.now().isoformat(timespec='minutes').replace(':', '-')
    save_folder = os.path.join(
        os.path.dirname(sys.argv[0]),  # Assuming this is the script's directory
        "outputs",
        "optimization-results",
        f"linear_regression-gemini-{datetime_str}/",
    )
    os.makedirs(save_folder, exist_ok=True)
    print(f"Result directory: {save_folder}")



    # Multiple repetitions of the experiment
    for i_rep in range(num_reps):
        print(f"\nRunning repetition {i_rep + 1}")

        # synthetic data for linear regression
        X = np.linspace(1, num_points, num_points)
        noise = np.random.randn(num_points) * 5  # Gaussian noise
        y = X * w_true + b_true + noise

        # Initialize parameters for optimization
        initial_w = np.random.uniform(low=5, high=25)
        initial_b = np.random.uniform(low=5, high=25)

        previous_attempts = [] # A list to keep track of previous attempts
        current_w = initial_w
        current_b = initial_b
        sol_w=current_w
        sol_b=current_b
        current_loss = evaluate_loss(X, y, current_w, current_b)
        sol_loss=current_loss
        threshold_loss = 1e-5  #threshold for convergence
        previous_attempts.append((current_w, current_b, current_loss))
        for _ in range(10):
          rand_w = np.random.uniform(low=5, high=25)
          rand_b = np.random.uniform(low=5, high=25)
          rand_loss = evaluate_loss(X, y, rand_w, rand_b)
          previous_attempts.append((rand_w, rand_b, rand_loss))
          if rand_loss<sol_loss:
              sol_w=rand_w
              sol_b=rand_b
              sol_loss=rand_loss

        for i_step in range(max_num_steps):
            # Prompt for the Gemini model

            prompt = create_prompt(current_w, current_b, X, y, previous_attempts)
            print(f"\nStep {i_step + 1} Prompt:\n{prompt}\n")
             # Ensure at least 4 seconds have passed since the last API call
            if last_api_call_time is not None:
                time_since_last_call = time.time() - last_api_call_time
                if time_since_last_call < 30:
                    time.sleep(30 - time_since_last_call)
            
            
            # Generate response from the model
            response = model.generate_content(prompt)
            last_api_call_time = time.time()  # Update the time after the API call
            response_text = response.text
            print(f"Model Response:\n{response_text}\n")
            
            try:
                new_w, new_b = parse_response(response_text)
            except ValueError as e:
                print(f"Error parsing response: {e}")
                continue  # Skip to the next iteration
            
            # Evaluate the loss with new parameters
            new_loss = evaluate_loss(X, y, new_w, new_b)
            print(f"New Parameters: w = {new_w}, b = {new_b}, Loss = {new_loss}")
            current_loss=new_loss
            current_w=new_w
            current_b=new_b
            
            # Update parameters if the new loss is lower
            if new_loss < sol_loss:
                sol_w = new_w
                sol_b = new_b
                sol_loss = new_loss
                print(f"Updated Parameters: w = {current_w}, b = {current_b}, Loss = {current_loss}")
            else:
                print("No improvement in loss.")

            # Add the attempt to the list of previous attempts
            previous_attempts.append((new_w, new_b, new_loss))

            # Stop if loss is below the threshold
            if current_loss < threshold_loss:
                print(f"Optimal parameters found: w = {current_w}, b = {current_b}, Loss = {current_loss}")
                break

        # Save the results of this repetition
        results = {
            'final_w': sol_w,
            'final_b': sol_b,
            'final_loss': sol_loss
        }
        result_path = os.path.join(save_folder, f"results_rep_{i_rep + 1}.json")
        with open(result_path, 'w') as file:
            json.dump(results, file, indent=4)

def create_prompt(w, b, X, y, previous_attempts):
    # Function definition
    function_definition = """
We are trying to minimize the loss function:
""".strip()

    # Include all previous (w, b) and loss, sorted in descending order of loss
    previous_attempts_section = ""
    if previous_attempts:
        # Sort the previous attempts in descending order based on loss
        sorted_attempts = sorted(previous_attempts, key=lambda x: x[2], reverse=True)
        previous_attempts_section = "Previous (w, b) pairs and their loss values (sorted by loss in descending order):\n"
        for idx, (prev_w, prev_b, prev_loss) in enumerate(sorted_attempts, 1):  # Include all attempts
            previous_attempts_section += f"{idx}. w = {prev_w:.0f}, b = {prev_b:.0f}, Loss = {prev_loss:.0f}\n"

    # Current parameters and loss
    current_loss = evaluate_loss(X, y, w, b)
    current_params_section = f"Current Parameters:\nw = {w:.0f}, b = {b:.0f}\nCurrent Loss: {current_loss:.0f}"

    # Instructions
    instructions = """
Please suggest a new (w, b) pair that is different from all pairs above and results in a lower loss value. Use the provided function definition and data points to calculate the loss. Do not include any explanations or code. The output should be exactly in the format:

[w, b]

For example:

[13, 15]
""".strip()

    # Construct the full prompt
    prompt = "\n\n".join([
        function_definition,
        previous_attempts_section,
        current_params_section,
        instructions
    ])

    return prompt


def parse_response(response_text):
    # Use regular expression to find numbers within square brackets
    match = re.search(r"\[([\d\.\-\s,]+)\]", response_text)
    if match:
        numbers = match.group(1)
        # Split the numbers by comma or whitespace
        parts = re.split(r"[,\s]+", numbers.strip())
        if len(parts) >= 2:
            try:
                new_w = float(parts[0])
                new_b = float(parts[1])
                return new_w, new_b
            except ValueError:
                pass
    # If parsing fails, raise an error or return None
    raise ValueError("Failed to parse w and b from the model's response.")

def evaluate_loss(X, y, w, b):
    predictions = X * w + b
    return np.sum((y - predictions) ** 2)

if __name__ == "__main__":
    app.run(main)

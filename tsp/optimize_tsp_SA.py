import datetime
import json
import numpy as np
import os
import sys
import re
import time
from absl import app
import google.generativeai as genai

def main(_):
    _GEMINI_API_KEY = '' 
    genai.configure(api_key=_GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-pro")

    # Optimization parameters
    num_points = 10         
    num_steps = 20           
    num_decimals = 0         
    num_starting_points = 10 
    num_decode_per_step = 1  

    # SA parameters
    initial_temperature = 1000
    cooling_rate = 0.9
    temperature = initial_temperature

    # Rate limiting parameters
    max_requests_per_minute = 3
    request_interval = 60 / max_requests_per_minute  # Time in seconds between requests
    last_api_call_time = None

    # Result directory
    datetime_str = datetime.datetime.now().isoformat(timespec='minutes').replace(':', '-')
    save_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),  # Assuming this is the script's directory
        "outputs",
        "optimization-results",
        f"tsp-gemini-{datetime_str}/",
    )
    os.makedirs(save_folder, exist_ok=True)
    print(f"Result directory: {save_folder}")

    # Generate random points for TSP
    x = [60, 180, 100, 140, 20, 100, 200, 140, 40, 100,
             180, 60, 120, 180, 20, 100, 200, 20, 60, 160]
    y= [200, 200, 180, 180, 160, 160, 160, 140, 120, 120,
             100, 80, 80, 60, 40, 40, 40, 20, 20, 20]
    x = np.round(x, num_decimals)
    y = np.round(y, num_decimals)

    # Initialize solutions
    initial_solutions = []
    while len(initial_solutions) < num_starting_points:
        sol = np.random.permutation(num_points)
        if sol[0] != 0:  # Ensure all traces start from point 0
            continue
        initial_solutions.append(sol.tolist())

    # Evaluate initial solutions
    previous_traces = []
    previous_trace_set = set()  # To keep track of unique traces
    best_initial_distance = None
    best_initial_trace = None
    for sol in initial_solutions:
        distance = evaluate_distance(x, y, sol)
        trace_str = ','.join(map(str, sol))
        previous_traces.append((trace_str, distance))
        previous_trace_set.add(trace_str)
        if best_initial_distance is None or distance < best_initial_distance:
            best_initial_distance = distance
            best_initial_trace = sol


    current_trace = best_initial_trace
    current_distance = best_initial_distance
    sol_trace = current_trace
    sol_distance = current_distance
    threshold_distance = 0.01  # Define a threshold for the acceptable solution

    # Optimization process
    total_steps = 0
    while total_steps < num_steps:
        total_steps += 1
        print(f"\nOptimization Step {total_steps}")


        prompt = create_prompt(x, y, previous_traces, num_points, current_trace, current_distance)
        print(f"\nPrompt:\n{prompt}\n")

        # Enforce rate limiting
        if last_api_call_time is not None:
            time_since_last_call = time.time() - last_api_call_time
            if time_since_last_call < request_interval:
                time.sleep(request_interval - time_since_last_call)


        try:
            response = model.generate_content(prompt)
            last_api_call_time = time.time()
            response_text = response.text
            print(f"Model Response:\n{response_text}\n")

            new_trace = parse_response(response_text, num_points)
            new_distance = evaluate_distance(x, y, new_trace)
            print(f"New Trace Distance: {new_distance}")
            trace_str = ','.join(map(str, new_trace))

            delta_distance = new_distance - current_distance

            if trace_str in previous_trace_set:
                print("Generated trace is a repeat. Not adding it to previous_traces or meta prompt.")
            else:
                if delta_distance < 0:
                    # Accept the new trace
                    current_trace = new_trace
                    current_distance = new_distance
                    previous_traces.append((trace_str, new_distance))
                    previous_trace_set.add(trace_str)

                    # Update the best solution if applicable
                    if new_distance < sol_distance:
                        sol_trace = new_trace
                        sol_distance = new_distance
                    print(f"Accepted new trace with improved distance: {current_distance}")

                    # Check if we've reached the threshold
                    if current_distance < threshold_distance:
                        print(f"Optimal trace found with distance: {current_distance}")
                        break

                    # Cool down the temperature
                    temperature *= cooling_rate
                    continue
                else:
                    # Calculate acceptance probability
                    acceptance_probability = np.exp(-delta_distance / temperature)
                    random_value = np.random.rand()
                    print(f"Acceptance Probability: {acceptance_probability}, Random Value: {random_value}")

                    if random_value < acceptance_probability:
                        # Accept worse solution due to SA
                        current_trace = new_trace
                        current_distance = new_distance
                        previous_traces.append((trace_str, new_distance))
                        previous_trace_set.add(trace_str)
                        print(f"Accepted worse trace due to SA with distance: {current_distance}")

                        # Cool down the temperature
                        temperature *= cooling_rate
                        continue
                    else:
                        print("Rejected LLM-generated trace. Proceeding to random generation step.")

            if current_distance < threshold_distance:
                print(f"Optimal trace found with distance: {current_distance}")
                break

        except ValueError as e:
            print(f"Error parsing response: {e}")
            continue  # Skip to the next iteration
        except genai.exceptions.APIError as api_err:
            print(f"API error occurred: {api_err}")
            time.sleep(5)  # Wait before retrying
            continue

    results = {
        'final_trace': sol_trace,
        'final_distance': sol_distance
    }
    result_path = os.path.join(save_folder, f"results.json")
    with open(result_path, 'w') as file:
        json.dump(results, file, indent=4)

def create_prompt(x, y, previous_traces, num_points, current_trace, current_distance):

    points_description = "We have the following points with their coordinates:\n"
    for idx, (xi, yi) in enumerate(zip(x, y)):
        points_description += f"{idx}: ({xi}, {yi})\n"


    previous_traces_section = ""
    if previous_traces:
        sorted_traces = sorted(previous_traces, key=lambda x: x[1])
        previous_traces_section = "Here are some previous traces and their total distances (shorter distances are better):\n"
        for idx, (trace_str, distance) in enumerate(sorted_traces, 1):
            previous_traces_section += f"{idx}. Trace: [{trace_str}], Distance: {distance:.2f}\n"


    current_trace_section = ""
    if current_trace is not None:
        current_trace_str = ','.join(map(str, current_trace))
        current_trace_section = f"\nCurrent Trace and its total distance:\nTrace: [{current_trace_str}], Distance: {current_distance:.2f}\n"


    instructions = f"""
Please provide a new trace different from previous traces that visits all {num_points} points exactly once, starting from point 0, and results in a shorter total distance than the previous traces. The trace should be a list of integers from 0 to {num_points - 1} in the format:

[0, 2, 1, 3, 4, ..., N]

Make sure to include each point exactly once and do not repeat any points. Do not include any explanations or additional text.
""".strip()


    prompt = "\n".join([
        points_description,
        previous_traces_section,
        current_trace_section,
        instructions
    ])

    return prompt

def parse_response(response_text, num_points):
    match = re.search(r"\[([^\]]+)\]", response_text)
    if match:
        numbers = match.group(1)
        # Split the numbers by comma or whitespace
        parts = re.split(r"[,\s]+", numbers.strip())
        trace = []
        for part in parts:
            part = part.strip()
            if part.isdigit():
                num = int(part)
                if 0 <= num < num_points:
                    trace.append(num)
                else:
                    raise ValueError(f"Number out of range: {num}")
            else:
                raise ValueError(f"Invalid number format: {part}")

        if len(trace) != num_points:
            raise ValueError(f"Trace does not contain {num_points} points.")
        if len(set(trace)) != num_points:
            raise ValueError("Trace contains duplicate points.")
        if trace[0] != 0:
            raise ValueError("Trace does not start with point 0.")
        return trace


    raise ValueError("Failed to find a trace in the model's response.")

def evaluate_distance(x, y, trace):
    distance = 0.0
    for i in range(len(trace) - 1):
        idx_current = trace[i]
        idx_next = trace[i + 1]
        dx = x[idx_next] - x[idx_current]
        dy = y[idx_next] - y[idx_current]
        distance += np.hypot(dx, dy)
    idx_last = trace[-1]
    idx_start = trace[0]
    dx = x[idx_start] - x[idx_last]
    dy = y[idx_start] - y[idx_last]
    distance += np.hypot(dx, dy)
    return round(distance, 2)

def generate_random_trace(num_points):
    trace = list(range(num_points))
    np.random.shuffle(trace[1:])  # Keep the first element as 0
    return trace

if __name__ == "__main__":
    app.run(main)

import json
import os

path = '/home/golikovp/Antigravity/Linguistic_traps/experiments/context_saturation/conv_results/tiiuae_Falcon-H1R-7B/HuggingFaceH4_aime_2024/tiiuae_Falcon-H1R-7B_HuggingFaceH4_aime_2024_s42_20260201_215708_CONVERSATION.json'

with open(path, 'r') as f:
    data = json.load(f)


print(f"Total entries: {len(data)}")

max_len = 0
max_agent = None
max_msg_idx = -1
found_agent_obj = None

for agent in data:
    history = agent.get('history_dump', [])
    for i, msg in enumerate(history):
        l = len(msg)
        if l > max_len:
            max_len = l
            max_agent = agent['id']
            max_msg_idx = i
            found_agent_obj = agent



print(f"Max Message Length: {max_len} chars")
print(f"Found in Agent ID: {max_agent}, Msg Index: {max_msg_idx}")

# Print stats for the ACTUAL max object found
if found_agent_obj:
    max_history = found_agent_obj.get('history_dump', [])
    lengths = [len(m) for m in max_history]
    print(f"Agent {max_agent} (TARGET) History Lengths: {lengths}")
    print(f"Total Chars: {sum(lengths)}")
    
    with open('massive_dump.json', 'w') as f:
        json.dump(found_agent_obj['history_dump'], f, indent=2)
    print("Dumped massive history to massive_dump.json")





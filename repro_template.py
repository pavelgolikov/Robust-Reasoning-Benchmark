import jinja2
import json

# The template provided by the user
TEMPLATE_STR = """{# --- System Prompt Handling --- #}
{%- if messages and messages[0]['role'] == 'system' %}
  {% set system_msg = messages[0]['content'] %}  
  {%- set remaining_messages = messages[1:] %}
{%- else %}
  {% set system_msg = "You are Falcon, a helpful AI assistant created by Technology Innovation Institute (TII). To answer the user's question, you first think about the reasoning process and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e., <think> reasoning process here </think> answer here." %}
  {%- set remaining_messages = messages %}
{%- endif %}
{%- if tools %}
<|im_start|>system
{{ system_msg }}
# Tools
You may call one or more functions to assist with the user query. You are provided with function signatures within <tools></tools> XML tags.
<tools>
{%- for tool in tools %}
{{- "" }}
{{ tool | tojson }}
{%- endfor %}
{{- "" }}
</tools>
For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
<|im_end|>
{%- else %}
<|im_start|>system
{{ system_msg }}
<|im_end|>
{%- endif %}
{# --- Render remaining messages --- #}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}{%- for message in remaining_messages %}
  {%- set content = message.get('content','') %}
  {%- if message['role'] == 'user' %}
    {{- '<|im_start|>' + message['role'] + '\n' + content + '<|im_end|>\n' }}
  {%- elif message['role'] == 'assistant' %}
    {{- '<|im_start|>' + message.role + '\n' }}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- content + '\n' }}
            {%- endif %}
        {%- else %}
            {{- content + '\n' }}
        {%- endif %}
    {%- if tools and message.tool_calls %}
      {%- for tool_call in message.tool_calls %}
          {%- if tool_call.function is defined %}
              {%- set tool_call = tool_call.function %}
          {%- endif %}
          {{-'<tool_call>\n' }}
          {{- '{"name": "'+ tool_call.name + '", "arguments":' }}
          {%- if tool_call.arguments is string -%}
          {{ tool_call.arguments }}
          {%- else -%}
          {{ tool_call.arguments | tojson }}
          {%- endif -%}
          {{- '}' }}
          {{- '\n</tool_call>\n' }}
      {%- endfor %}
    {%- endif %}
    {%- if not loop.last %}
      {{- '<|im_end|>' + '\n' }}
    {%- else %}
      {{- '<|im_end|>' }}
    {%- endif %}
  {%- elif message['role'] == 'tool' %}
    {# Tool responses treated as user messages #}
    {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
        {{- '<|im_start|>user' }}
    {%- endif %}
    {{- '\n<tool_response>\n' + message['content'] + '\n</tool_response>' }}
    {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
        {{- '<|im_end|>\n' }}
    {%- endif %}
  {%- endif %}
  {# --- Add generation prompt after last message if requested --- #}
  {%- if loop.last and add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
  {%- endif %}
{%- endfor %}"""

def main():
    print("Loading history dump...")
    with open('history_dump.txt', 'r') as f:
        history_content = json.load(f)

    # Reconstruct message list [System, User, Asst, User, Asst, User]
    # Assuming standard roles sequence from history_dump analysis
    # Msg 0: System (105 chars)
    # Msg 1: User (4540 chars)
    # Msg 2: Asst (30982 chars)
    # Msg 3: User (4716 chars)
    # Msg 4: Asst (18136 chars)
    # Msg 5: User (4719 chars)
    
    roles = ["system", "user", "assistant", "user", "assistant", "user"]
    messages = []
    
    for i, content in enumerate(history_content):
        role = roles[i] if i < len(roles) else "user"
        messages.append({"role": role, "content": content})
        
    print(f"Prepared {len(messages)} messages.")
    
    # Setup Jinja Environment
    env = jinja2.Environment()
    template = env.from_string(TEMPLATE_STR)
    
    # Render
    print("Rendering...")
    try:
        output = template.render(messages=messages, tools=None, add_generation_prompt=True)
        print(f"Render Success.")
        print(f"Output Length: {len(output)} chars")
        
        # Check for duplication
        if len(output) > 200000:
            print("EXPLOSION DETECTED!")
        else:
            print("Output size is normal.")
            
    except Exception as e:
        print(f"Render Error: {e}")

if __name__ == "__main__":
    main()

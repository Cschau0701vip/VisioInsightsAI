import streamlit as st
import os
import base64
from openai import OpenAI

from scripts.make_prediction import process_and_visualize_data
import glob

output_dir = 'output'

token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-4o-mini"
# model_name = "Phi-4"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

def get_file_data_url(file: str, file_format: str) -> str:
    """
    Helper function to converts an file to a data URL string.

    Args:
        file (str): The path to the file.
        file_format (str): The format of the file.

    Returns:
        str: The data URL of the file.
    """
    try:
        with open(file, "rb") as f:
            file_data = base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Could not read '{file}'.")
        exit()
    return f"data:image/{file_format};base64,{file_data}"

# Streamlit app title
st.title("AI Insights: Data & Image Analysis")
st.write("This app uses OpenAI's GPT-4 model to provide insights on data and image analysis.")

# Toggle button to select mode
mode = st.radio(
    "Select Mode:",
    ["Chat Bot Mode", "Advanced Mode"],
    index=0,
    horizontal=True
)

# Initialize uploaded_file and uploaded_data_file to None
uploaded_file = None
uploaded_data_file = None

# Text box for user to input a question
question_input = st.text_area(
    "Enter your question here: *",
    placeholder="Type your question about the data prediction or image analysis...",
    value=st.session_state.get("selected_option", ""),
    key="question_input"
)
    
if mode == "Advanced Mode":
    # Show file uploader and dropdown elements based on role
    st.write("Upload your files and select graph type in Advanced Mode.")
    # Update session state with role options if not already set
    if "role_options" not in st.session_state:
        st.session_state["role_options"] = ["Admin", "Dealer", "User"]

    if "role_options" in st.session_state:
        selected_option = st.radio(
            "Select your role:",
            st.session_state["role_options"],
            index=0,
            horizontal=True
        )
        role_input = selected_option
        
    if role_input == "Admin":          
        # Admin can upload both image and data files, and select graph type
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        st.write("OR")
        uploaded_data_file = st.file_uploader("Upload your CSV or JSON data file: *", type=["csv", "json"])
        graph_input_type = st.selectbox(
            "Select the type for the graph to be generated:",
            ["None", "Pie (Default)", "Linear", "Scatter", "Residual", "Correlation"],
        )
    elif role_input == "Dealer":
        # Dealer can only upload data files
        uploaded_data_file = st.file_uploader("Upload your CSV or JSON data file: *", type=["csv", "json"])
        graph_input_type = st.selectbox(
            "Select the type for the graph to be generated:",
            ["None", "Pie (Default)", "Linear", "Scatter", "Residual", "Correlation"],
        )
    elif role_input == "User":
        # User can only upload image files
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    
    # Store the uploaded file in session state
    if uploaded_data_file is not None:
        st.session_state["uploaded_data_file"] = uploaded_data_file
    if uploaded_file is not None:
        st.session_state["uploaded_file"] = uploaded_file
        
# Submit button
if st.button("Submit"):
    if question_input:
        # Call the OpenAI API to get the explanation
        try:
            if uploaded_file is not None:
                print("Uploaded file")
                # Save the uploaded image temporarily
                temp_image_path = "temp_image." + uploaded_file.name.split(".")[-1]
                with open(temp_image_path, "wb") as f:
                    f.write(uploaded_file.read())
                
                # Get the image data URL
                image_data_url = get_file_data_url(temp_image_path, uploaded_file.type.split("/")[-1])
                
                # Call the OpenAI API
                response = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system", 
                            "content": question_input
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": question_input},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_data_url,
                                        "detail": "low"
                                    },
                                },
                            ],
                        },
                    ],
                    model=model_name,
                    stream=True,
                    stream_options={'include_usage': True}
                )
            elif uploaded_data_file is not None:
                print("Uploaded data file")
                print(uploaded_data_file.name)
                # Save the uploaded file temporarily
                temp_file_path = "temp_file." + uploaded_data_file.name.split(".")[-1]
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_data_file.read())
                
                print(temp_file_path)
                # Get the image data URL
                # file_data_url = get_file_data_url(temp_file_path, uploaded_file.type.split("/")[-1])
                process_and_visualize_data(temp_file_path, output_dir)
                graph_type_map = {
                    "Pie (Default)": "pie",
                    "Linear": "linear",
                    "Scatter": "scatter",
                    "Residual": "residual",
                    "Correlation": "correlation"
                }
                selected_graph_type = graph_type_map.get(graph_input_type, "pie")
                # Find the first matching file for the selected graph type
                matching_files = glob.glob(os.path.join(output_dir, f"{selected_graph_type}*.png"))
                if matching_files:
                    output_file = matching_files[0]
                else:
                    raise FileNotFoundError(f"No matching output file found for graph type '{selected_graph_type}'.")
                
                # Get the image data URL
                output_image_url = get_file_data_url(output_file, "png")
                
                # Call the OpenAI API
                response = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": question_input,
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": question_input,
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": output_image_url,
                                        "detail": "low"
                                    },
                                },
                            ],
                        },
                    ],
                    model=model_name,
                    stream=True,
                    stream_options={'include_usage': True}
                )
            else:
                 # Call the OpenAI API
                response = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": question_input,
                        }
                    ],
                    model=model_name,
                    stream=True,
                    stream_options={'include_usage': True}
                )
                # st.warning("Please upload an image before submitting.")

            explanation = ""
            usage = None
            for update in response:
                if hasattr(update, "choices") and update.choices:
                    if hasattr(update.choices[0], "delta") and hasattr(update.choices[0].delta, "content"):
                        explanation += update.choices[0].delta.content or ""
                if hasattr(update, "usage"):
                    usage = update.usage

            if usage:
                print("\n")
                for k, v in usage.model_dump().items():
                    print(f"{k} = {v}")
            # If an image was uploaded, display it
            if uploaded_file is not None:
                st.subheader("Uploaded Image")
                st.image(temp_image_path, caption="Uploaded Image", use_container_width=True)
            # Display the explanation
            st.subheader("Explanation")
            st.write(explanation)
            # Display the saved plot in the Streamlit UI, If an data was uploaded, display it
            if uploaded_data_file is not None:
                st.subheader("Predicted Output")
                graph_captions = {
                    "pie": "Distribution of Vehicle Conditions",
                    "linear": "Linear Relationship Analysis",
                    "scatter": "Scatter Plot of Data",
                    "residual": "Residual Analysis",
                    "correlation": "Correlation Analysis"
                }
                dynamic_caption = graph_captions.get(selected_graph_type, "Generated Graph")
                st.image(output_file, caption=dynamic_caption, use_container_width=True)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question before submitting.")
        
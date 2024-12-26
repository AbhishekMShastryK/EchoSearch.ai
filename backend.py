import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from openai import OpenAI
from secret import OPENAI_API_KEY
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import timedelta

app = Flask(__name__)
CORS(app)

def generate_embeddings(text):
        client = OpenAI(
            api_key=OPENAI_API_KEY,
        )
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        norm = np.linalg.norm(embedding)
        normalized_embedding = (embedding / norm).tolist() if norm != 0 else embedding.tolist()
        return normalized_embedding

def summarize_text(combined_text, whole_video_summary, student_query, course_name=" "):
    client = OpenAI(
        api_key=OPENAI_API_KEY,
    )
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system", 
                "content": (
                    "You are a teaching assistant for the course {course_name}. When responding to students' questions, "
                    "your goal is to directly address their query in a clear, concise, and detailed way. Use the specific part of the lecture "
                    "as your primary focus, but also incorporate the overall lecture context to ensure the explanation aligns with the course themes. "
                    "Tie the explanation to the key learning objectives and practical applications covered in the course, helping the student "
                    "understand not just what the professor said, but also how it connects to the course material."
                ).format(course_name=course_name)
            },
            {
                "role": "user", 
                "content": (
                    f"The student asked the following question:\n\n"
                    f"'{student_query}'\n\n"
                    f"Here's what the professor explained during the specific part of the lecture:\n\n"
                    f"{combined_text}\n\n"
                    f"And here's what the overall lecture was about for additional context:\n\n"
                    f"{whole_video_summary}\n\n"
                    "Please answer the student's question in a concise, detailed, and approachable way. "
                    "Highlight its relevance to the course material and connect it to the concepts or skills the professor is teaching."
                )
            }
        ]
    )

    return response.choices[0].message.content

@app.route('/search', methods=['POST'])
def search_similar_records():
    data = request.json
    query = data.get('query')
    top_k = 3
    neighbor_count = 2
    collection_name = "gaits_lecture_data_collection"

    if not query:
        return jsonify({"error": "Query is required."}), 400

    # Generate embedding for the query
    query_embedding = generate_embeddings(query)
    if query_embedding is None:
        return jsonify({"error": "Failed to generate query embedding."}), 500

    # Load the Milvus collection
    try:
        connections.connect(host='127.0.0.1', port='19530')
        collection = Collection(name=collection_name)
        collection.load()
    except Exception as e:
        return jsonify({"error": f"Failed to load collection: {str(e)}"}), 500

    # Perform the initial search
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}  # IP: Inner Product for Cosine Similarity
    try:
        results = collection.search(
            data=[query_embedding],  # The query embedding
            anns_field="lecture_embedding",  # The field to search
            param=search_params,
            limit=top_k,  # Number of top results
            output_fields=["id", "lecture_name", "video_name", "start_time", "lecture_summary", "whole_video_summary"]  # Include ID in results
        )
    except Exception as e:
        return jsonify({"error": f"Search failed: {str(e)}"}), 500

    if(results == []):
        return jsonify({"error": f"Search failed: {str(e)}"})
    # Parse results to get the top IDs and video names
    top_records = []
    for hits in results:
        for hit in hits:
            if hit.score < 0.3:  # Check similarity score threshold
                continue
            top_records.append({
                "id": hit.id,
                "lecture_name": hit.entity.get("lecture_name"),
                "video_name": hit.entity.get("video_name"),
                "start_time": hit.entity.get("start_time"),
                "lecture_summary": hit.entity.get("lecture_summary"),
                "whole_video_summary": hit.entity.get("whole_video_summary"),
                "score": hit.score
            })
            
    # Check if no results are returned
    if not results or not any(hits for hits in results):
        return jsonify({"message": "No results found."}), 404
    
    # Combine text and neighbors for each center_id
    video_combined_texts = {}
    for record in top_records:
        center_id = record["id"]
        lecture_name = record["lecture_name"]
        video_name = record["video_name"]
        center_start_time = record["start_time"]
        whole_video_summary = record["whole_video_summary"]
        neighbor_ids = list(range(center_id - neighbor_count, center_id + neighbor_count + 1))

        # Query neighbors within the same video
        expr = f"id in {neighbor_ids} && video_name == '{video_name}'"
        try:
            neighbors = collection.query(
                expr=expr,
                output_fields=["id", "video_name", "start_time", "lecture_summary", "whole_video_summary"],
                consistency_level="Strong"
            )
        except Exception as e:
            return jsonify({"error": f"Query for neighbors failed: {str(e)}"}), 500

        # Combine text from neighbors and the center record
        combined_text = " ".join([n["lecture_summary"] or "" for n in neighbors if "lecture_summary" in n])
        combined_text = (record["lecture_summary"] or "") + " " + combined_text

        if video_name not in video_combined_texts:
            video_combined_texts[video_name] = {
                "combined_text": combined_text,
                "start_time_list": [center_start_time],
                "whole_video_summary": whole_video_summary
            }
        else:
            video_combined_texts[video_name]["combined_text"] += " " + combined_text
            video_combined_texts[video_name]["start_time_list"].append(center_start_time)

    # Summarize combined text for each video
    final_results = []
    for video_name, content in video_combined_texts.items():
        min_start_time = min(content["start_time_list"])
        
        # Filter and convert non-minimum start times to HH:MM
        non_min_start_times = [
            str(timedelta(seconds=start_time))
            for start_time in content["start_time_list"]
            if start_time != min_start_time
        ]
        
        # Summarize the combined text
        summarized_text = summarize_text(content["combined_text"], content["whole_video_summary"], query, lecture_name)

        final_results.append({
            "video_name": video_name,
            "start_time": str(timedelta(seconds=min_start_time)),
            "non_min_start_times": non_min_start_times,
            "summary": summarized_text
        })
    return jsonify(final_results)

if __name__ == '__main__':
    app.run(debug=True)

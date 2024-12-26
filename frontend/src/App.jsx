import { useState, useEffect } from "react";
import "./App.css";

function App() {
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false); // Loading state
  const [queries, setQueries] = useState(() => {
    const savedQueries = localStorage.getItem("queries");
    return savedQueries ? JSON.parse(savedQueries) : [];
  });
  const [videos, setVideos] = useState(() => {
    const savedVideos = localStorage.getItem("videos");
    return savedVideos ? JSON.parse(savedVideos) : []; // Load videos from localStorage
  });
  const [summaries, setSummaries] = useState(() => {
    const savedSummaries = localStorage.getItem("summaries");
    return savedSummaries ? JSON.parse(savedSummaries) : []; // Load summaries from localStorage
  });
  const [isQueryLogMinimized, setIsQueryLogMinimized] = useState(false); // Toggle for query log
  const [isInitial, setIsInitial] = useState(false); // To track initial state

  // Save queries to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem("queries", JSON.stringify(queries));
  }, [queries]);

  useEffect(() => {
    localStorage.setItem("videos", JSON.stringify(videos));
  }, [videos]);

  useEffect(() => {
    localStorage.setItem("summaries", JSON.stringify(summaries));
  }, [summaries]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (query.trim()) {
      setIsInitial(false); // Mark as no longer in the initial state
      setIsLoading(true); // Start loading animation
      setVideos([]); // Clear existing videos
      setSummaries([]); // Clear existing summaries
      const now = new Date();
      const time = now.toLocaleTimeString("en-GB", {
        hour: "2-digit",
        minute: "2-digit",
      });
      const date = now.toLocaleDateString("en-GB", {
        day: "2-digit",
        month: "2-digit",
        year: "numeric",
      });

      setQueries((prevQueries) => [
        ...prevQueries,
        { text: query, time, date },
      ]);
      setQuery(""); // Clear the input field

      try {
        // Call the Flask API with the user's query
        const response = await fetch("http://127.0.0.1:5000/search", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ query }),
        });

        if (response.ok) {
          const data = await response.json();
          console.log("API Response:", data); // Log the response

          const mappedVideos = data.map((item) => ({
            video_name: item.video_name || "Unknown Video",
            start_time: item.start_time || "00:00",
            non_min_start_times: item.non_min_start_times || "",
          }));
          const mappedSummaries = data.map((item) => ({
            video_name: item.video_name || "Unknown Video",
            start_time: item.start_time || "00:00",
            summary: item.summary || "No summary available",
          }));

          setVideos(mappedVideos);
          setSummaries(mappedSummaries);

          if (mappedVideos.length === 0 && mappedSummaries.length === 0) {
            alert("No results found for your query.");
          }
        } else {
          setVideos([]);
          setSummaries([]);
          alert("No results found for your query.");
        }
      } catch (error) {
        setVideos([]);
        setSummaries([]);
        alert("No results found for your query.");
      } finally {
        setIsLoading(false); // Stop loading animation
      }
    }
  };

  const groupByDate = () => {
    const grouped = {};
    queries.forEach((q) => {
      if (!grouped[q.date]) {
        grouped[q.date] = [];
      }
      grouped[q.date].push(q);
    });
    return grouped;
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const clearQueries = () => {
    setQueries([]);
    localStorage.removeItem("queries"); // Clear from localStorage
  };

  const toggleQueryLog = () => {
    setIsQueryLogMinimized((prevState) => !prevState);
  };

  const groupedQueries = groupByDate();

  return (
    <>
      <div
        className="header"
        onClick={() => setIsInitial(true)}
        style={{ cursor: "pointer" }}
      >
        <span className="header-black">EchoSearch</span>
        <span className="header-orange">.ai</span>
      </div>

      {isLoading && <div className="loading-animation">Loading...</div>}

      <div className={`query-log ${isQueryLogMinimized ? "minimized" : ""}`}>
        <div className="query-log-header">
          <h3>Recent Queries</h3>
          <div className="query-log-controls">
            <button onClick={toggleQueryLog}>
              {isQueryLogMinimized ? "Expand" : "Minimize"}
            </button>
            <button onClick={clearQueries}>Clear</button>
          </div>
        </div>
        {!isQueryLogMinimized && (
          <div className="query-content">
            {Object.keys(groupedQueries)
              .reverse() // Reverse the order of dates
              .map((date, index) => (
                <div key={index} className="date-group">
                  <h4 className="query-date">{date}</h4>
                  <ul>
                    {groupedQueries[date]
                      .slice() // Copy the array to avoid mutating the original
                      .reverse() // Reverse the order of prompts for this date
                      .map((q, idx) => (
                        <li key={idx} className="query-item">
                          <span className="query-text">{q.text}</span>
                          <span className="query-time">{q.time}</span>
                        </li>
                      ))}
                  </ul>
                </div>
              ))}
          </div>
        )}
      </div>
      {(videos.length === 0 && summaries.length === 0) || isInitial ? (
        <div className="centered-input">
          <form onSubmit={handleSubmit}>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Enter your query..."
            />
            <button type="submit">
              <i className="fas fa-arrow-right"></i>
            </button>
          </form>
        </div>
      ) : (
        <div className="container">
          <div className="content">
            <div className="videos-section">
              <h2>Videos</h2>
              <div className="videos-content">
                {videos.length > 0 ? (
                  videos.map((video, index) => {
                    const videoPath = `/lecture_videos/${video.video_name}.mp4`;
                    const startSeconds = video.start_time
                      .split(":")
                      .reduce((acc, time) => 60 * acc + +time, 0);

                    return (
                      <div key={index} className="video-container">
                        <p className="video-name">{video.video_name}</p>
                        {video.non_min_start_times &&
                          video.non_min_start_times.length > 0 && (
                            <p className="additional-start-times">
                              Also check at:{" "}
                              {video.non_min_start_times.join(", ")}
                            </p>
                          )}
                        <video
                          controls
                          width="100%"
                          src={videoPath}
                          onLoadedMetadata={(e) => {
                            e.target.currentTime = startSeconds; // Set playback start time
                          }}
                        >
                          Your browser does not support the video tag.
                        </video>
                      </div>
                    );
                  })
                ) : (
                  <p className="no-content">No videos available</p>
                )}
              </div>
            </div>

            <div className="summary-section">
              <h2>Summaries</h2>
              <div className="summary-content">
                {summaries.length > 0 ? (
                  summaries.map((summary, index) => (
                    <div key={index} className="summary-container">
                      <p className="video-name">{summary.video_name}</p>
                      <p>{summary.summary}</p>
                    </div>
                  ))
                ) : (
                  <p className="no-content">No summaries available</p>
                )}
              </div>
            </div>
          </div>
          <div className="search-container">
            <form onSubmit={handleSubmit}>
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Enter your query..."
              />
              <button type="submit">
                <i className="fas fa-arrow-right"></i>
              </button>
            </form>
          </div>
        </div>
      )}
      <div className="footer">
        EchoSearch.ai can make mistakes. Kindly verify the information provided.
      </div>
    </>
  );
}

export default App;

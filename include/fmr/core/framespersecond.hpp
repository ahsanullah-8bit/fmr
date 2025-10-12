#pragma once

#include <chrono>
#include <mutex>
#include <shared_mutex>
#include <vector>

namespace fmr {
    class frames_per_second {
        using time_point = std::chrono::time_point<std::chrono::steady_clock>;
    public:
        explicit frames_per_second(int max_frames = 1000, int last_n_seconds = 10)
            : m_start(std::chrono::time_point<std::chrono::steady_clock>()),
            m_max_frames(max_frames),
            m_last_nsecs(last_n_seconds) {
        }

        void start() {
            std::unique_lock<std::shared_mutex> lock(m_mtx);
            m_start = std::chrono::steady_clock::now();
        }

        void update() {
            std::unique_lock<std::shared_mutex> lock(m_mtx);
            auto now = std::chrono::steady_clock::now();
            if (m_start.time_since_epoch().count() == 0.0) {
                // Initialize the first time
                m_start = now;
            }

            m_timestamps.emplace_back(now);
            if (m_timestamps.size() > m_max_frames + 100) {
                // Truncate the list when it goes 100 over the max_size.
                m_timestamps.erase(m_timestamps.begin() + (m_timestamps.size() - m_max_frames), m_timestamps.end());
            }

            expire_timestamps(now);
        }

        double fps() {
            std::shared_lock<std::shared_mutex> lock(m_mtx);
            auto now = std::chrono::steady_clock::now();
            if (m_start.time_since_epoch().count() == 0.0) {
                m_start = now;
            }

            // Compute the (approximate) frames in the last n seconds
            expire_timestamps(now);
            auto seconds = std::min(std::chrono::duration_cast<std::chrono::seconds>(now - m_start).count(), m_last_nsecs.count());
            // Avoid divide by zero
            if (seconds == 0.0) {
                seconds = 1.0;
            }

            return m_timestamps.size() / seconds;
        }

    private:
        // Remove aged out timestamps
        void expire_timestamps(time_point now) {
            time_point threshold = now - m_last_nsecs;
            while (!m_timestamps.empty() && m_timestamps.front() < threshold) {
                m_timestamps.erase(m_timestamps.begin());
            }
        }

        time_point m_start;
        int m_max_frames;
        std::chrono::duration<long long> m_last_nsecs;
        std::vector<time_point> m_timestamps;
        std::shared_mutex m_mtx;
    };
}

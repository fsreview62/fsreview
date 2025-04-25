#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <thread>
#include <signal.h>
#include <bits/stdc++.h>
#include <errno.h>
#include <sys/wait.h>

using namespace std;
using namespace chrono;

vector<pair<string, pair<string, string>>> files_and_users = {
    {"../../", {"downsampled_data_10000_rows_first_arrivals.csv", "1379"}},
    // {"../../", {"downsampled_data_100000_first_arrivals.csv", "7445"}},
    // {"../../", {"downsampled_data_90k_3kusers.csv", "3200"}},
    // {"../", {"downsampled_data_10000_120_seconds.csv", "2000"}},
    // {"../", {"downsampled_data_10000_600_seconds.csv", "3247"}},
    // {"../", {"downsampled_data_10000_1800_seconds.csv", "4564"}},
    // {"../", {"downsampled_data_10000_3600_seconds.csv", "5354"}},
    // {"../", {"downsampled_data_10000_first_users.csv", "113"}},
    // {"../", {"downsampled_data_10000_first_interactions.csv", "1339"}},
};

vector<string> schedulers = {
    "vtc_fair",
    "lshare_fair",
    // "fs_fair_interaction_limit_expect",
    // "fs_fair_wsc_expect"
};

vector<string> fs_ratelimits = {
    "10",
    "50",
    "100",
    "500"
};

vector<string> app_limits = {
    // "200",
    "2000"
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cout << "Usage: ./a.out <app_stat>\n";
        return 0;
    }
    string app_stat(argv[1]);
    vector<pair<string, string>> commands;
    for(pair<string, pair<string, string>> file_and_users: files_and_users) {
        string root_path = file_and_users.first;
        string rel_path = file_and_users.second.first;
        string filename = root_path + rel_path;
        string users = file_and_users.second.second;
        for(string app_limit: app_limits) {
            for(string scheduler: schedulers) {
                if ((scheduler[0] == 'f' && scheduler[1] == 's') || scheduler == "lshare_fair") {
                    for (string ratelimit: fs_ratelimits) {
                        string output_file = root_path + "outputs/100000_requests/" + rel_path + "___" + scheduler + "___" + ratelimit + "___" + app_limit + "___" + app_stat + ".jsonl";
                        string request_log_name = root_path + "outputs/100000_requests/" + rel_path + "___" + scheduler + "___" + ratelimit + "___" + app_limit + "___" + app_stat + ".csv";
                        string cmd1 = "python3 launch_server.py --model-setting S1 --num-adapter " + users + " --num-token 100000 --scheduler " + scheduler + " --rate-limit " + ratelimit;
                        if (scheduler == "lshare_fair") {
                            cmd1 += " --enable-abort";
                        }
                        string cmd2 = "python3 run_exp.py --model-setting S1 --suite real_m365_trace --input " + filename + "  --output " + output_file + " --app-limit " + app_limit + " --request-log " + request_log_name + " --app-stat " + app_stat;
                        string log_file = root_path + "outputs/100000_requests/" + rel_path + "___" + scheduler + "___" + ratelimit + "___" + app_limit + "___" + app_stat + ".log";
                        cmd1 += " > " + log_file;
                        commands.push_back({cmd1, cmd2});
                    }
                } else {
                    string output_file = root_path + "outputs/100000_requests/" + rel_path + "___" + scheduler + "___" + app_limit + "___" + app_stat + ".jsonl";
                    string request_log_name = root_path + "outputs/100000_requests/" + rel_path + "___" + scheduler + "___" + app_limit + "___" + app_stat + ".csv";
                    string cmd1 = "python3 launch_server.py --model-setting S1 --num-adapter " + users + " --num-token 100000 --scheduler " + scheduler;
                    string cmd2 = "python3 run_exp.py --model-setting S1 --suite real_m365_trace --input " + filename + "  --output " + output_file + " --app-limit " + app_limit + " --request-log " + request_log_name + " --app-stat " + app_stat;
                    string log_file = root_path + "outputs/100000_requests/" + rel_path + "___" + scheduler + "___" + app_limit + "___" + app_stat + ".log";
                    cmd1 += " > " + log_file;
                    commands.push_back({cmd1, cmd2});
                }
            }
        }
    }
    chdir("./fair_bench");
    for(pair<string, string> cmd_pair: commands) {
        cout << cmd_pair.first << "\n" << cmd_pair.second << "\n\n";
        pid_t pid = fork();
        if(pid < 0) {
            cout << "FORK FAILED??\n";
        } else if(pid == 0) {
            pid_t myPid = getpid();
            setpgid(myPid, myPid);
            cout << "Child process group id: " << getpgid(0) << "\n";
            ofstream fout;
            fout.open("running_server.sh");
            fout << cmd_pair.first;
            fout.close();
            char *argv[] = {"sh", "running_server.sh", NULL};
            execvp("sh", argv);
            return 0;
        } else {
            cout << "Child pid: " << pid << "\n";
            this_thread::sleep_for(chrono::seconds(120));
            cout << "Hopefully server is up and running by now?\n";
            pid_t pid2 = fork();
            if (pid2 < 0) {
                cout << "SECOND FORK FAILED??\n";
            }
            else if (pid2 == 0) {
                ofstream fout;
                fout.open("running_experiment.sh");
                fout << cmd_pair.second;
                fout.close();
                char *argv[] = {"sh", "running_experiment.sh", NULL};
                execvp("sh", argv);
            }
            int returnStatus;    
            auto t1 = high_resolution_clock::now();
            waitpid(pid2, &returnStatus, 0);
            auto t2 = high_resolution_clock::now();
            auto ms_int = duration_cast<milliseconds>(t2 - t1);
            cout << "Experiment finished\n";
            cout << "Total time: " << ms_int.count() << " miliseconds\n";
            int retkill = kill(-pid, SIGTERM);
            if(retkill == -1) {
                cout << "ERRNO: " << errno;
            }
            cout << "Process killed with signal: " << retkill << "\n";
            this_thread::sleep_for(chrono::seconds(10));
        }
    }
    return 0;
}
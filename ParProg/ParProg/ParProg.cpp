#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <chrono>
#include <atomic>
#include <omp.h>
#include <iomanip>
#include <fstream>
#include <Windows.h>

// Реализация на основе шаблона Taskbag 
class TaskBag {
private:
    std::queue<int> tasks; // Очередь задач (индексы строк)
    std::mutex queue_mutex; // Синхронизация доступа к очереди
    std::atomic<bool> finished; // флаг завершения всех задач
    int matrix_size; // Размер матрицы
    const std::vector<std::vector<int>>& A; // Ссылка на матрицу A
    const std::vector<std::vector<int>>& B; // Ссылка на матрицу B
    std::vector<std::vector<int>>& C; // Ссылка на результирующую матрицу C

public:
    // Конструктор инициализирует все необходимые данные
    TaskBag(int size, const std::vector<std::vector<int>>& matA,
        const std::vector<std::vector<int>>& matB,
        std::vector<std::vector<int>>& matC)
        : matrix_size(size), A(matA), B(matB), C(matC), finished(false) {

        // Заполняем очередь задач индексами строк
        for (int i = 0; i < size; ++i) {
            tasks.push(i);
        }
    }

    // Получение задачи из очереди
    bool get_task(int& task) {
        std::lock_guard<std::mutex> lock(queue_mutex);
        if (tasks.empty()) {
            return false;
        }
        task = tasks.front();
        tasks.pop();
        return true;
    }

    // Выполнение задачи - умножение строки матрицы
    void execute_task(int row) {
        for (int j = 0; j < matrix_size; ++j) {
            C[row][j] = 0;
            for (int k = 0; k < matrix_size; ++k) {
                C[row][j] += A[row][k] * B[k][j];
            }
        }
    }

    // Установка флага завершения
    void set_finished() {
        finished = true;
    }

    // Проверка, завершены ли все задачи
    bool is_finished() {
        return finished && tasks.empty();
    }
};

// Функция рабочего потока
void worker_thread(TaskBag& task_bag) {
    int task;
    while (!task_bag.is_finished()) {
        if (task_bag.get_task(task)) {
            // Если получили задачу - выполняем её
            task_bag.execute_task(task);
        }
        else {
            // Если задач нет - отдаем управление другим потокам
            std::this_thread::yield();
        }
    }
}

// Умножение матрис с использованием Taskbag
std::vector<std::vector<int>> matrix_multiply_taskbag(int size,
    const std::vector<std::vector<int>>& A,
    const std::vector<std::vector<int>>& B,
    int num_threads) {
    std::vector<std::vector<int>> C(size, std::vector<int>(size, 0));

    TaskBag task_bag(size, A, B, C);

    std::vector<std::thread> threads;
    // Создаем рабочие потоки
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker_thread, std::ref(task_bag));
    }

    // Устанавливаем флаг завершения
    task_bag.set_finished();

    // Ожидаем завершения всех потоков
    for (auto& thread : threads) {
        thread.join();
    }

    return C;
}

// Реализация умножения матриц с использованием OpenMP
std::vector<std::vector<int>> matrix_multiply_openmp(int size,
    const std::vector<std::vector<int>>& A,
    const std::vector<std::vector<int>>& B) {
    std::vector<std::vector<int>> C(size, std::vector<int>(size, 0));

    // Параллельный цикл с динамическим распределением итераций
#pragma omp parallel for //schedule(dynamic)
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            int sum = 0;
            for (int k = 0; k < size; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }

    return C;
}

// Реализация умножения матриц с использованием OpenMP
std::vector<std::vector<int>> matrix_multiply_simple(int size,
    const std::vector<std::vector<int>>& A,
    const std::vector<std::vector<int>>& B) {
    std::vector<std::vector<int>> C(size, std::vector<int>(size, 0));

    // Параллельный цикл с динамическим распределением итераций
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            int sum = 0;
            for (int k = 0; k < size; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }

    return C;
}

// Генерация случайной матрицы
std::vector<std::vector<int>> generate_random_matrix(int size) {
    std::vector<std::vector<int>> matrix(size, std::vector<int>(size));
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i][j] = rand() % 100; // Заполняем случайными числами от 0 до 99
        }
    }
    return matrix;
}

// Проверка корректности результатов
bool verify_results(const std::vector<std::vector<int>>& C1,
    const std::vector<std::vector<int>>& C2) {
    if (C1.size() != C2.size()) return false;
    for (size_t i = 0; i < C1.size(); ++i) {
        for (size_t j = 0; j < C1[i].size(); ++j) {
            if (C1[i][j] != C2[i][j]) {
                return false;
            }
        }
    }
    return true;
}

// Функция проведения экспериментов производительности
void performance_experiment() {
    std::vector<int> matrix_sizes = { 100, 200, 500, 800, 1000 }; // Размеры матриц для тестирования
    std::vector<int> thread_counts = { 2, 4, 8 }; // Количество потоков для тестирования
    int num_runs = 5; // Количество запусков для усреднения результатов

    // Заголовок таблицы результатов
    std::cout << "Размер матрицы | Потоки | Время Taskbag (мс) | Время OpenMP (мс) | Ускорение Taskbag | Ускорение OpenMP | Эффективность Taskbag | Эффективность OpenMP\n";
    std::cout << "---------------|--------|-------------------|-------------------|-------------------|------------------|-----------------------|----------------------\n";

    // Создание файла для сохранения данных
    std::ofstream data_file("performance_data.csv");
    // Использование точки с запятой как разделителя
    data_file << "Matrix Size;Threads;Taskbag Time (ms);OpenMP Time (ms);Speedup Taskbag;Speedup OpenMP;Efficiency Taskbag;Efficiency OpenMP\n";

    for (int size : matrix_sizes) {
        auto A = generate_random_matrix(size);
        auto B = generate_random_matrix(size);

        // Измерение времени последовательной версии для сравнения
        auto start_seq = std::chrono::high_resolution_clock::now();
        auto C_seq = matrix_multiply_simple(size, A, B);
        auto end_seq = std::chrono::high_resolution_clock::now();
        double seq_time = std::chrono::duration<double, std::milli>(end_seq - start_seq).count();

        for (int threads : thread_counts) {
            double total_taskbag_time = 0;
            double total_openmp_time = 0;

            // Многократный запуск для усреднения результатов
            for (int run = 0; run < num_runs; ++run) {
                // Тестирование реализации на Taskbag
                auto start_taskbag = std::chrono::high_resolution_clock::now();
                auto C_taskbag = matrix_multiply_taskbag(size, A, B, threads);
                auto end_taskbag = std::chrono::high_resolution_clock::now();
                total_taskbag_time += std::chrono::duration<double, std::milli>(end_taskbag - start_taskbag).count();

                // Тестирование реализации на OpenMP
                omp_set_num_threads(threads);
                auto start_openmp = std::chrono::high_resolution_clock::now();
                auto C_openmp = matrix_multiply_openmp(size, A, B);
                auto end_openmp = std::chrono::high_resolution_clock::now();
                total_openmp_time += std::chrono::duration<double, std::milli>(end_openmp - start_openmp).count();

                // Проверка корректности результатов
                if (!verify_results(C_seq, C_taskbag) || !verify_results(C_seq, C_openmp)) {
                    std::cout << "ОШИБКА: Результаты не совпадают!\n";
                }
            }

            // Расчет средних значений
            double avg_taskbag_time = total_taskbag_time / num_runs;
            double avg_openmp_time = total_openmp_time / num_runs;

            // Расчет ускорения
            double speedup_taskbag = seq_time / avg_taskbag_time;
            double speedup_openmp = seq_time / avg_openmp_time;

            // Расчет эффективности
            double efficiency_taskbag = speedup_taskbag / threads;
            double efficiency_openmp = speedup_openmp / threads;

            // Вывод результатов в консоль
            std::cout << std::setw(14) << size << " | "
                << std::setw(6) << threads << " | "
                << std::setw(17) << std::fixed << std::setprecision(2) << avg_taskbag_time << " | "
                << std::setw(17) << avg_openmp_time << " | "
                << std::setw(17) << std::setprecision(2) << speedup_taskbag << " | "
                << std::setw(16) << speedup_openmp << " | "
                << std::setw(21) << std::setprecision(3) << efficiency_taskbag << " | "
                << std::setw(18) << efficiency_openmp << "\n";



            data_file << size << ";"
                << threads << ";"
                << avg_taskbag_time << ";"
                << avg_openmp_time << ";"
                << speedup_taskbag << ";"
                << speedup_openmp << ";"
                << efficiency_taskbag << ";"
                << efficiency_openmp << "\n";
        }
    }

    data_file.close();
    std::cout << "\nДанные сохранены в файл performance_data.csv\n";
    std::cout << "Каждое значение находится в отдельной ячейке Excel\n";
}

int main() {
    SetConsoleOutputCP(1251);
    std::cout << "Анализ производительности параллельного умножения матриц\n";
    std::cout << "========================================================\n\n";

    performance_experiment();

    return 0;
}
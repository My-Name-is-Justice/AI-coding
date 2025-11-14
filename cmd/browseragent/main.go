package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"math"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/joho/godotenv"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/polzovatel/ai-agent-for-browser/internal/agent"
	"github.com/polzovatel/ai-agent-for-browser/internal/agent/planner"
	"github.com/polzovatel/ai-agent-for-browser/internal/browser"
	"github.com/polzovatel/ai-agent-for-browser/internal/llm"
	"github.com/polzovatel/ai-agent-for-browser/internal/memory"
	"github.com/polzovatel/ai-agent-for-browser/internal/tools"
)

type cliOptions struct {
	task          string
	storageState  string
	saveStatePath string
	temperature   float64
}

// Пример запуска:
//
//	./browseragent -storage storage/hh_session.json
//
// После запуска вводите команду start и задачу в интерактивном режиме.
func main() {
	opts := parseFlags()

	if opts.task == "" {
		task, cancelled, err := promptTaskFromConsole()
		if err != nil {
			log.Fatal().Err(err).Msg("не удалось получить описание задачи")
		}
		if cancelled {
			fmt.Println("Выполнение отменено пользователем.")
			return
		}
		opts.task = task
	}

	if err := godotenv.Load(); err != nil && !os.IsNotExist(err) {
		log.Fatal().Err(err).Msg("failed to load .env file")
	}

	setupLogger()

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	if err := run(ctx, opts); err != nil {
		log.Fatal().Err(err).Msg("agent execution failed")
	}
}

// parseFlags читает аргументы запуска и извлекает описание задачи.
func parseFlags() cliOptions {
	task := flag.String("task", "", "Task description for the browser agent")
	storage := flag.String("storage", "", "Path to Playwright storage state for authenticated sessions")
	saveState := flag.String("save-state", "", "Path to save updated storage state after run")
	temperature := flag.Float64("temperature", math.NaN(), "LLM temperature (0.0-1.0). Если не указано, используется значение из окружения или значение по умолчанию")
	flag.Parse()

	return cliOptions{
		task:          strings.TrimSpace(*task),
		storageState:  strings.TrimSpace(*storage),
		saveStatePath: strings.TrimSpace(*saveState),
		temperature:   *temperature,
	}
}

// setupLogger настраивает формат вывода логов для удобного чтения в терминале.
func setupLogger() {
	zerolog.TimeFieldFormat = time.RFC3339Nano
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr, TimeFormat: time.RFC3339Nano})
}

// run собирает зависимости и запускает оркестратор с указанной задачей.
func run(ctx context.Context, opts cliOptions) error {
	// Создаём клиента LLM по данным окружения (ключ и модель).
	llmClient, err := llm.NewClientFromEnv()
	if err != nil {
		return fmt.Errorf("create llm client: %w", err)
	}

	// При необходимости переопределяем температуру модели.
	if !math.IsNaN(opts.temperature) {
		// Температура фиксируется в глобальной переменной планировщика.
		planner.SetTemperature(float32(opts.temperature))
	}

	// Инициализируем память, с которой агент будет работать в течение задачи.
	mem := memory.NewConversationMemory()

	// Запускаем браузер и получаем контроллер страницы.
	pwLauncher, err := browser.NewLauncher(ctx)
	if err != nil {
		return fmt.Errorf("start browser: %w", err)
	}
	// Обязательно закрываем браузер по завершении run.
	defer pwLauncher.Close()

	ctrl, err := pwLauncher.NewControllerWithState(ctx, opts.storageState)
	if err != nil {
		return fmt.Errorf("create browser controller: %w", err)
	}
	defer ctrl.Close(ctx)

	// Собираем планировщик и набор инструментов для взаимодействия с браузером.
	plannerEngine := planner.NewPlanAndExecute(llmClient)
	toolbox := tools.NewStandardToolbox(ctrl, terminalPrompter())

	// Конструируем оркестратор с заданными лимитами шагов и таймаутом.
	taskID := memory.GenerateStepID()
	orchestratorLogger := log.With().
		Str("task_id", taskID).
		Str("component", "orchestrator").
		Logger()
	orch := agent.NewOrchestrator(agent.Config{
		MaxSteps:    150,
		IdleTimeout: 3 * time.Minute,
	}, llmClient, plannerEngine, mem, toolbox, orchestratorLogger)

	// Передаём оркестратору информацию о задаче и запускаем выполнение.
	taskInfo := agent.Task{
		ID:          taskID,
		Description: opts.task,
	}
	runErr := orch.Run(ctx, taskInfo)
	if opts.saveStatePath != "" && runErr == nil {
		// После успешного выполнения можно сохранить обновлённую сессию браузера.
		if err := ctrl.SaveState(ctx, opts.saveStatePath); err != nil {
			log.Error().Err(err).Str("path", opts.saveStatePath).Msg("не удалось сохранить storage state")
		} else {
			log.Info().Str("path", opts.saveStatePath).Msg("storage state сохранён")
		}
	}
	printRunSummary(taskInfo, mem, runErr)
	return runErr
}

// printRunSummary выводит в консоль историю выполнения задачи.
func printRunSummary(task agent.Task, mem memory.Memory, runErr error) {
	fmt.Println("\n==================== ИТОГИ ====================")
	fmt.Printf("Задача: %s\n", task.Description)
	fmt.Printf("ID задачи: %s\n", task.ID)
	if runErr != nil {
		fmt.Printf("Статус: ❌ %v\n", runErr)
	} else {
		fmt.Println("Статус: ✅ Завершено успешно")
	}

	// Сохраняем шаги, чтобы вывести количество и финальное резюме.
	steps := mem.Steps()
	fmt.Printf("Всего шагов: %d\n", len(steps))
	if len(steps) > 0 {
		// Последний шаг обычно содержит итоговое сообщение модели.
		last := steps[len(steps)-1]
		if last.ActionName == "finish" && strings.TrimSpace(last.Observation) != "" {
			fmt.Printf("Итоговое резюме: %s\n", last.Observation)
		}
	}

	fmt.Println("===============================================")
	fmt.Println()
}

func terminalPrompter() tools.PromptFunc {
	reader := bufio.NewReader(os.Stdin)
	return func(ctx context.Context, message string) (string, error) {
		fmt.Println("\n=== Требуется ввод пользователя ===")
		fmt.Println(message)
		fmt.Print("> ")

		// Ожидаем, пока пользователь введёт ответ и нажмёт Enter.
		input, err := reader.ReadString('\n')
		if err != nil {
			return "", fmt.Errorf("ошибка чтения ввода: %w", err)
		}

		select {
		case <-ctx.Done():
			return "", ctx.Err()
		default:
		}

		return strings.TrimSpace(input), nil
	}
}

func promptTaskFromConsole() (task string, cancelled bool, err error) {
	reader := bufio.NewReader(os.Stdin)
	fmt.Println("Доступные команды: start — начать выполнение задачи, exit — завершить программу.")
	for {
		fmt.Print("Введите команду (start/exit): ")
		cmd, readErr := reader.ReadString('\n')
		if readErr != nil {
			return "", false, readErr
		}
		cmd = strings.TrimSpace(cmd)
		switch strings.ToLower(cmd) {
		case "exit":
			// Пользователь явно отказался запускать агента.
			return "", true, nil
		case "start":
			// После команды start просим описать задачу.
			for {
				fmt.Print("Опишите задачу: ")
				line, readErr := reader.ReadString('\n')
				if readErr != nil {
					return "", false, readErr
				}
				task = strings.TrimSpace(line)
				if task == "" {
					fmt.Println("Описание задачи не может быть пустым.")
					continue
				}
				return task, false, nil
			}
		default:
			// Предупреждаем о неверной команде и возвращаемся в начало цикла.
			if cmd != "" {
				fmt.Println("Неизвестная команда. Используйте start или exit.")
			}
		}
	}
}

package agent

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/rs/zerolog"

	"github.com/polzovatel/ai-agent-for-browser/internal/agent/planner"
	"github.com/polzovatel/ai-agent-for-browser/internal/llm"
	"github.com/polzovatel/ai-agent-for-browser/internal/memory"
	"github.com/polzovatel/ai-agent-for-browser/internal/tools"
)

var (
	ErrStepLimitExceeded = errors.New("агент исчерпал лимит шагов и завершил работу")
)

const (
	historyWindow   = 10
	logFieldMaxRune = 200
)

// Config описывает основные настройки оркестратора.
type Config struct {
	MaxSteps    int
	IdleTimeout time.Duration
}

// Task описывает цель, которую агент должен выполнить.
type Task struct {
	ID          string
	Description string
}

// Orchestrator координирует взаимодействие планировщика, инструментов и памяти.
type Orchestrator struct {
	config  Config
	llm     llm.Client
	planner planner.Engine
	memory  memory.Memory
	tools   tools.Toolbox
	logger  zerolog.Logger
}

// NewOrchestrator связывает ключевые подсистемы в единую точку управления.
func NewOrchestrator(cfg Config, llmClient llm.Client, plannerEngine planner.Engine, mem memory.Memory, toolbox tools.Toolbox, logger zerolog.Logger) *Orchestrator {
	return &Orchestrator{
		config:  cfg,
		llm:     llmClient,
		planner: plannerEngine,
		memory:  mem,
		tools:   toolbox,
		logger:  logger,
	}
}

// Run запускает выполнение задачи. Детальная логика появится на следующих шагах.
func (o *Orchestrator) Run(ctx context.Context, task Task) error {
	// Сбрасываем память, чтобы убедиться, что прошлые задачи не оставили контекст.
	o.memory.Reset(task.ID, task.Description)

	// Сразу выводим исходное задание в "чат".
	fmt.Printf("\nuser: %s\n", task.Description)

	idleTimeout := o.config.IdleTimeout

	for step := 1; step <= o.config.MaxSteps; step++ {
		// Проверяем, не отменил ли пользователь выполнение (Ctrl+C) или не истёк ли контекст.
		if err := ctx.Err(); err != nil {
			return err
		}

		// Формируем описание текущего шага для LLM.
		state := planner.State{
			Step:            step,
			TaskDescription: task.Description,
			History:         convertHistory(o.memory.RecentSteps(historyWindow)),
			Tools:           o.tools.Describe(),
		}

		// Просим планировщик (LLM) решить, что делать дальше.
		decision, err := o.planner.Next(ctx, state)
		if err != nil {
			return fmt.Errorf("планировщик не смог сформировать следующий шаг: %w", err)
		}

		// Если планировщик решил завершить работу — сохраняем финальный шаг и выходим.
		if decision.Finish {
			o.memory.AddStep(memory.Step{
				Index:       step,
				Thought:     decision.Thought,
				ActionName:  "finish",
				Observation: decision.FinalSummary,
			})
			if summary := strings.TrimSpace(decision.FinalSummary); summary != "" {
				fmt.Printf("agent: %s\n", summary)
			}
			return nil
		}

		// Если инструмент не выбран, фиксируем пропуск шага.
		if decision.ActionName == "" {
			o.memory.AddStep(memory.Step{
				Index:       step,
				Thought:     decision.Thought,
				ActionName:  "noop",
				Observation: "Планировщик не выбрал действие",
			})
			o.logger.Info().
				Int("step", step).
				Msg("шаг пропущен: действие не выбрано")
			continue
		}

		actionCtx := ctx
		var cancel context.CancelFunc
		if idleTimeout > 0 {
			// Ограничиваем выполнение инструмента таймаутом, чтобы агент не зависал навсегда.
			actionCtx, cancel = context.WithTimeout(ctx, idleTimeout)
		}

		// Выполняем инструмент и фиксируем результат.
		result, err := o.tools.Invoke(actionCtx, decision.ActionName, decision.ActionInput)
		if cancel != nil {
			cancel()
		}
		observation := result.Observation
		if err != nil {
			observation = fmt.Sprintf("ошибка при выполнении %s: %v", decision.ActionName, err)
			o.logger.Warn().
				Int("step", step).
				Err(err).
				Str("action", decision.ActionName).
				Msg("ошибка при выполнении действия")
		}

		// Сохраняем завершённый шаг в памяти.
		o.memory.AddStep(memory.Step{
			Index:       step,
			Thought:     decision.Thought,
			ActionName:  decision.ActionName,
			ActionInput: decision.ActionInput,
			Observation: observation,
		})

		// Выводим краткую запись в консоль, чтобы пользователь видел прогресс.
		fmt.Printf("agent[%d]: %s -> %s\n", step, decision.ActionName, formatObservation(decision.ActionName, observation))
	}

	return ErrStepLimitExceeded
}

// convertHistory преобразует историю памяти в формат, удобный планировщику.
func convertHistory(steps []memory.Step) []planner.HistoryItem {
	result := make([]planner.HistoryItem, 0, len(steps))
	for _, step := range steps {
		result = append(result, planner.HistoryItem{
			Step:        step.Index,
			Thought:     step.Thought,
			ActionName:  step.ActionName,
			ActionInput: step.ActionInput,
			Observation: step.Observation,
		})
	}
	return result
}

// truncateForLog ограничивает длину текстов, попадающих в лог.
func truncateForLog(text string) string {
	trimmed := strings.TrimSpace(strings.ReplaceAll(text, "\n", " "))
	runes := []rune(trimmed)
	if len(runes) <= logFieldMaxRune {
		return trimmed
	}
	return string(runes[:logFieldMaxRune]) + "..."
}

// formatObservation сворачивает "шумные" наблюдения, чтобы не перегружать лог.
func formatObservation(action, observation string) string {
	obs := strings.TrimSpace(observation)
	if obs == "" {
		return "-"
	}
	if strings.HasPrefix(obs, "ошибка") {
		// Ошибки показываем полностью, чтобы не потерять контекст.
		return obs
	}
	switch action {
	case "read_page", "collect_texts", "collect_links":
		// Эти инструменты часто возвращают очень большие фрагменты — заменяем пометкой.
		return "(данные опущены)"
	default:
		return truncateForLog(obs)
	}
}

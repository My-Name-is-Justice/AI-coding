package memory

import (
	"strconv"
	"sync"
	"time"
)

// Step описывает завершённый шаг агента: что подумал, сделал и наблюдал.
type Step struct {
	Index       int
	Thought     string
	ActionName  string
	ActionInput map[string]any
	Observation string
	CreatedAt   time.Time
}

// Memory описывает интерфейс хранилища контекста между шагами.
type Memory interface {
	Reset(taskID string, description string)
	AddStep(step Step)
	Steps() []Step
	RecentSteps(limit int) []Step
	TaskID() string
	TaskDescription() string
}

// ConversationMemory — потокобезопасная реализация памяти агента.
type ConversationMemory struct {
	mu          sync.RWMutex
	taskID      string
	description string
	steps       []Step
}

// NewConversationMemory создаёт экземпляр памяти, готовый к работе.
func NewConversationMemory() *ConversationMemory {
	return &ConversationMemory{}
}

// Reset очищает память и записывает информацию о новой задаче.
func (m *ConversationMemory) Reset(taskID string, description string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.taskID = taskID
	m.description = description
	m.steps = m.steps[:0]
}

// AddStep сохраняет завершённый шаг с наблюдениями.
func (m *ConversationMemory) AddStep(step Step) {
	m.mu.Lock()
	defer m.mu.Unlock()
	step.CreatedAt = time.Now()
	m.steps = append(m.steps, step)
}

// Steps возвращает снимок истории шагов.
func (m *ConversationMemory) Steps() []Step {
	m.mu.RLock()
	defer m.mu.RUnlock()
	result := make([]Step, len(m.steps))
	copy(result, m.steps)
	return result
}

// RecentSteps возвращает последние limit шагов (или все, если limit <= 0).
func (m *ConversationMemory) RecentSteps(limit int) []Step {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if limit <= 0 || limit >= len(m.steps) {
		result := make([]Step, len(m.steps))
		copy(result, m.steps)
		return result
	}

	start := len(m.steps) - limit
	result := make([]Step, limit)
	copy(result, m.steps[start:])
	return result
}

// TaskID возвращает идентификатор текущей задачи.
func (m *ConversationMemory) TaskID() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.taskID
}

// TaskDescription возвращает описание задачи.
func (m *ConversationMemory) TaskDescription() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.description
}

// GenerateStepID генерирует простой уникальный идентификатор запуска задачи.
func GenerateStepID() string {
	return strconv.FormatInt(time.Now().UnixNano(), 36)
}

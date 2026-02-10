package markov

import (
	"math"
	"testing"
)

func approxEqual(a, b float64) bool {
	return math.Abs(a-b) < 0.001
}

func TestRecordAndProbability(t *testing.T) {
	c := New()
	c.Record("A", "B")
	c.Record("A", "B")
	c.Record("A", "B")
	c.Record("A", "C")

	if !approxEqual(c.Probability("A", "B"), 0.75) {
		t.Errorf("P(B|A) = %f, want 0.75", c.Probability("A", "B"))
	}
	if !approxEqual(c.Probability("A", "C"), 0.25) {
		t.Errorf("P(C|A) = %f, want 0.25", c.Probability("A", "C"))
	}
}

func TestProbabilityEmpty(t *testing.T) {
	c := New()
	if c.Probability("A", "B") != 0 {
		t.Error("empty chain should return 0")
	}
	if c.Probability("", "B") != 0 {
		t.Error("empty from should return 0")
	}
	if c.Probability("A", "") != 0 {
		t.Error("empty to should return 0")
	}
}

func TestRecordEmptyStrings(t *testing.T) {
	c := New()
	c.Record("", "B")
	c.Record("A", "")
	c.Record("", "")
	if c.TransitionCount() != 0 {
		t.Error("empty strings should not be recorded")
	}
}

func TestPredict(t *testing.T) {
	c := New()
	c.Record("A", "B")
	c.Record("A", "B")
	c.Record("A", "B")
	c.Record("A", "C")

	if c.Predict("A") != "B" {
		t.Errorf("Predict(A) = %q, want B", c.Predict("A"))
	}
}

func TestPredictEmpty(t *testing.T) {
	c := New()
	if c.Predict("A") != "" {
		t.Error("empty chain predict should return empty string")
	}
}

func TestSelfTransition(t *testing.T) {
	c := New()
	c.Record("A", "A")
	c.Record("A", "A")

	if !approxEqual(c.Probability("A", "A"), 1.0) {
		t.Errorf("P(A|A) = %f, want 1.0", c.Probability("A", "A"))
	}
}

func TestPruneTopic(t *testing.T) {
	c := New()
	c.Record("A", "B")
	c.Record("A", "B")
	c.Record("A", "C")
	c.Record("B", "A")
	c.Record("B", "C")
	c.LastTopic = "B"

	c.PruneTopic("B")

	// B removed as destination from A
	if c.Probability("A", "B") != 0 {
		t.Error("P(B|A) should be 0 after pruning B")
	}
	// A→C should now be P=1.0 (only remaining transition from A)
	if !approxEqual(c.Probability("A", "C"), 1.0) {
		t.Errorf("P(C|A) = %f, want 1.0 after pruning B", c.Probability("A", "C"))
	}
	// B's outgoing transitions should be gone
	if c.Probability("B", "A") != 0 {
		t.Error("B's outgoing transitions should be removed")
	}
	if c.Probability("B", "C") != 0 {
		t.Error("B's outgoing transitions should be removed")
	}
	// LastTopic should be cleared
	if c.LastTopic != "" {
		t.Errorf("LastTopic = %q, want empty after pruning B", c.LastTopic)
	}
}

func TestPruneNonexistent(t *testing.T) {
	c := New()
	c.Record("A", "B")
	c.PruneTopic("Z") // should not panic
	if !approxEqual(c.Probability("A", "B"), 1.0) {
		t.Error("pruning nonexistent topic should not affect existing data")
	}
}

func TestTopTransitions(t *testing.T) {
	c := New()
	c.Record("A", "B")
	c.Record("A", "B")
	c.Record("A", "B")
	c.Record("A", "C")
	c.Record("A", "D")

	top := c.TopTransitions("A", 2)
	if len(top) != 2 {
		t.Fatalf("TopTransitions returned %d, want 2", len(top))
	}
	if top[0].TopicID != "B" {
		t.Errorf("top[0] = %q, want B", top[0].TopicID)
	}
	if !approxEqual(top[0].Probability, 0.6) {
		t.Errorf("top[0].Probability = %f, want 0.6", top[0].Probability)
	}
}

func TestTopTransitionsEmpty(t *testing.T) {
	c := New()
	if c.TopTransitions("A", 3) != nil {
		t.Error("empty chain TopTransitions should return nil")
	}
}

func TestTopTransitionsMoreThanExist(t *testing.T) {
	c := New()
	c.Record("A", "B")

	top := c.TopTransitions("A", 5)
	if len(top) != 1 {
		t.Errorf("TopTransitions returned %d, want 1", len(top))
	}
}

func TestTransitionCount(t *testing.T) {
	c := New()
	c.Record("A", "B")
	c.Record("A", "C")
	c.Record("B", "A")

	if c.TransitionCount() != 3 {
		t.Errorf("TransitionCount = %d, want 3", c.TransitionCount())
	}
}

package rbm

import (
	"bytes"
	"reflect"
	"testing"
)

func TestClassifierTrain(t *testing.T) {
	c := NewClassifier(2, 2, 3)
	input := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	output := []int{
		1,
		0,
		0,
		0,
	}
	c.Train(input, output, &Option{
		BatchSize: 10,
		Iteration: 2000,
		GibbsStep: 10,
	})

	for _, test := range []struct {
		in   []float64
		want int
	}{
		{
			in:   []float64{1, 1},
			want: 0,
		},
		{
			in:   []float64{0, 0},
			want: 1,
		},
		{
			in:   []float64{1, 0},
			want: 0,
		},
		{
			in:   []float64{0, 1},
			want: 0,
		},
	} {
		got := c.Classify(test.in)
		if !reflect.DeepEqual(got, test.want) {
			t.Logf("expect %v, got %v", test.want, got)
		}
	}
}

func TestClassifierMarshal(t *testing.T) {
	c := NewClassifier(4, 3, 2)
	buf := new(bytes.Buffer)
	err := c.WriteTo(buf)
	if err != nil {
		t.Fatal(err)
	}
	c2 := NewClassifier(4, 3, 2)
	err = c2.ReadFrom(buf)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(c, c2) {
		t.Fatalf("not equal")
	}
}

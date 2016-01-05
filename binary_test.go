package rbm

import (
	"bytes"
	"reflect"
	"testing"
)

func TestBinaryTrain(t *testing.T) {
	m := New(4, 3)
	data := [][]float64{
		{0, 0, 1, 1},
		{1, 1, 0, 0},
	}
	m.Train(data, &Option{
		BatchSize: 10,
		Iteration: 3000,
		GibbsStep: 10,
	})

	total := 1000
	for _, test := range []struct {
		in   []float64
		want []float64
	}{
		{
			in:   []float64{1, 1, 0, 0},
			want: []float64{1, 1, 0, 0},
		},
		{
			in:   []float64{0, 0, 1, 1},
			want: []float64{0, 0, 1, 1},
		},
		{
			in:   []float64{1, 1, 0, 1},
			want: []float64{1, 1, 0, 0},
		},
		{
			in:   []float64{1, 0, 1, 1},
			want: []float64{0, 0, 1, 1},
		},
	} {
		var count int
		for i := 0; i < total; i++ {
			got, _ := m.Reconstruct(test.in, 2)
			if !reflect.DeepEqual(got, test.want) {
				count++
			}
		}
		errRate := float64(count) / float64(total)
		if errRate > 0.05 {
			t.Fatalf("reconstruct error rate %f", errRate)
		}
	}
}

func TestBinaryMarshal(t *testing.T) {
	m := New(3, 2)
	buf := new(bytes.Buffer)
	err := m.WriteTo(buf)
	if err != nil {
		t.Fatal(err)
	}
	m2 := New(3, 2)
	err = m2.ReadFrom(buf)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(m, m2) {
		t.Fatalf("not equal")
	}
}

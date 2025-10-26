use core::fmt;
use std::ops;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct Cell(usize);

const H: Cell = Cell(0);

#[derive(Debug)]
struct Link {
    prev: Cell,
    next: Cell,
}

#[derive(Default, Debug)]
struct LinkedList {
    data: Vec<Link>,
}

impl ops::Index<Cell> for LinkedList {
    type Output = Link;

    fn index(&self, index: Cell) -> &Link {
        &self.data[index.0]
    }
}

impl ops::IndexMut<Cell> for LinkedList {
    fn index_mut(&mut self, index: Cell) -> &mut Link {
        &mut self.data[index.0]
    }
}

impl LinkedList {
    fn with_capacity(cap: usize) -> LinkedList {
        LinkedList {
            data: Vec::with_capacity(cap),
        }
    }

    fn alloc(&mut self) -> Cell {
        let cell = Cell(self.data.len());
        self.data.push(Link {
            prev: cell,
            next: cell,
        });
        cell
    }

    // Insert b after a
    fn insert(&mut self, a: Cell, b: Cell) {
        let c = self[a].next;

        // place here so failure happens earliest
        self[b].next = c;
        self[b].prev = a;

        self[a].next = b;
        self[c].prev = b;
    }

    // Remove b from its place
    fn remove(&mut self, b: Cell) {
        let a = self[b].prev;
        let c = self[b].next;

        self[a].next = c;
        self[c].prev = a;
    }

    // Place b back in place
    fn restore(&mut self, b: Cell) {
        let a = self[b].prev;
        let c = self[b].next;

        self[a].next = b;
        self[c].prev = b;
    }

    fn cursor(&self, head: Cell) -> Cursor {
        Cursor {
            head: head,
            current: head,
        }
    }
}

// Cursor for iterating through our linked list
struct Cursor {
    head: Cell,
    current: Cell,
}

impl Cursor {
    // The next (optional) item in the linked list
    fn next(&mut self, list: &LinkedList) -> Option<Cell> {
        self.current = list[self.current].next;
        if self.current == self.head {
            return None;
        }
        Some(self.current)
    }

    fn prev(&mut self, list: &LinkedList) -> Option<Cell> {
        self.current = list[self.current].prev;
        if self.current == self.head {
            return None;
        }
        Some(self.current)
    }
}

#[derive(Debug)]
pub struct Matrix {
    row_links: LinkedList,
    column_links: LinkedList,
    heads: Vec<Cell>,
    sizes: Vec<u32>,
    row_offsets: Vec<usize>,
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Sizes
        write!(f, "sizes: ")?;
        for s in &self.sizes {
            write!(f, "{:^7}", s)?;
        }
        writeln!(f)?;

        // Headers
        write!(f, "heads: ")?;
        for &Cell(c) in &self.heads {
            write!(f, "{:^7}", c.saturating_sub(1))?;
        }
        writeln!(f)?;

        // Rows
        write!(f, "row-links:  ")?;
        for link in &self.row_links.data {
            write!(f, " {:>2}|{:<2} ", link.prev.0, link.next.0)?;
        }
        writeln!(f)?;

        // Columns
        write!(f, "column-links:  ")?;
        for link in &self.column_links.data {
            write!(f, " {:>2}|{:<2} ", link.prev.0, link.next.0)?;
        }
        writeln!(f)?;

        // Indices
        write!(f, "indexes: ")?;
        for i in 00..self.row_links.data.len() {
            write!(f, "{:^7}", i)?;
        }
        writeln!(f)?;

        Ok(())
    }
}

impl Matrix {
    pub fn new(n_cols: usize) -> Self {
        let mut res = Matrix {
            sizes: Vec::with_capacity(n_cols + 1),
            heads: Vec::with_capacity(n_cols + 1),
            row_links: LinkedList::with_capacity(n_cols + 1),
            column_links: LinkedList::with_capacity(n_cols + 1),
            row_offsets: vec![0; n_cols + 1],
        };
        res.alloc_column();
        for _ in 0..n_cols {
            res.add_column();
        }
        res
    }

    fn add_column(&mut self) {
        let new_col = self.alloc_column();
        // insert on top rows row
        self.row_links.insert(self.row_links[H].prev, new_col);
    }

    fn alloc_column(&mut self) -> Cell {
        let cell = self.alloc(H);
        // replace with created cell
        self.heads[cell.0] = cell;
        // push 0 size
        self.sizes.push(0);
        cell
    }

    // create cell
    fn alloc(&mut self, head: Cell) -> Cell {
        // push head onto column vec
        self.heads.push(head);
        // create cell on rows vec and cols vec. should be identical.
        let cell = self.row_links.alloc();
        self.column_links.alloc();
        cell
    }

    pub fn add_row(&mut self, row: &[usize]) {
        let mut head: Cell;
        let mut prev = None;

        let row_offset = self.column_links.data.len();

        for &e in row {
            self.sizes[e + 1] += 1;
            head = Cell(e + 1);
            let new_cell = self.alloc(head);
            self.column_links
                .insert(self.column_links[head].prev, new_cell);

            if let Some(prev) = prev {
                self.row_links.insert(prev, new_cell);
            }
            prev = Some(new_cell);

            self.row_offsets.push(row_offset);
        }
    }

    pub fn row(&self, offset: usize) -> Vec<usize> {
        let c = Cell(offset);
        let mut cursor = self.row_links.cursor(c);

        let mut row: Vec<usize> = vec![self.heads[offset].0];

        while let Some(i) = cursor.next(&self.row_links) {
            row.push(self.heads[i.0].0);
        }
        row
    }

    // unlink the column from the matrix
    fn cover(&mut self, c: Cell) {
        // remove the header cell from the row links.
        self.row_links.remove(c);

        let mut ccursor = self.column_links.cursor(c);

        // go through the c-column
        while let Some(i) = ccursor.next(&self.column_links) {
            let mut rcursor = self.row_links.cursor(i);
            // row all i-row cells from columns only
            while let Some(j) = rcursor.next(&self.row_links) {
                self.column_links.remove(j);
                self.sizes[self.heads[j.0].0] -= 1;
            }
        }
    }

    // relink the column to the matrix
    fn uncover(&mut self, c: Cell) {
        let mut ccursor = self.column_links.cursor(c);
        while let Some(i) = ccursor.prev(&self.column_links) {
            let mut rcursor = self.row_links.cursor(i);
            while let Some(j) = rcursor.prev(&self.row_links) {
                self.column_links.restore(j);
                self.sizes[self.heads[j.0].0] += 1;
            }
        }
        self.row_links.restore(c);
    }

    pub fn solve(&mut self) -> Vec<Vec<usize>> {
        let mut partial = Vec::new();
        let mut solutions = Vec::new();
        self._solve(&mut partial, &mut solutions);
        solutions
    }

    fn _solve(&mut self, partial: &mut Vec<usize>, solutions: &mut Vec<Vec<usize>>) {
        // find the smallest/shortest column
        let column = {
            let mut row_cursor = self.row_links.cursor(H);
            let mut column = match row_cursor.next(&self.row_links) {
                Some(cell) => cell,
                None => {
                    // we found a solution!
                    let mut solution: Vec<usize> =
                        partial.iter().map(|&n| self.row_offsets[n]).collect();
                    solution.sort();
                    solutions.push(solution);
                    // leave
                    return;
                }
            };
            while let Some(next_column) = row_cursor.next(&self.row_links) {
                if self.sizes[next_column.0] < self.sizes[column.0] {
                    column = next_column;
                }
            }
            column
        };

        // cover the shortest column
        self.cover(column);

        // take column and use each row as a solution.
        let mut column_cursor = self.column_links.cursor(column);
        while let Some(cell_column) = column_cursor.next(&self.column_links) {
            // take cell as possible answer.
            partial.push(cell_column.0);
            // cover whole row of given column index
            let mut row_cursor = self.row_links.cursor(cell_column);
            while let Some(cell_row) = row_cursor.next(&self.row_links) {
                self.cover(self.heads[cell_row.0]);
            }

            // recurse
            self._solve(partial, solutions);

            // uncover whole row of given column index.
            let mut row_cursor = self.row_links.cursor(cell_column);
            while let Some(cell_row) = row_cursor.prev(&self.row_links) {
                self.uncover(self.heads[cell_row.0])
            }

            // remove cell from possible answer
            partial.pop();
        }

        self.uncover(column);
    }
}

#[derive(Clone)]
enum Step {
    CoverColumn,
    UncoverColumn,
    CoverRow,
    UncoverRow,
    Solution,
    Done,
}

pub struct IterativeSolver {
    matrix: Matrix,
    partial: Vec<usize>,
    columns: Vec<Cell>,
    cursors: Vec<Cursor>,
    step: Step,
}

impl IterativeSolver {
    pub fn new(matrix: Matrix) -> Self {
        IterativeSolver {
            matrix: matrix,
            partial: Vec::new(),
            columns: Vec::new(),
            cursors: Vec::new(),
            step: Step::CoverColumn,
        }
    }

    pub fn next(&mut self) -> Option<Vec<usize>> {
        loop {
            self.step = self._step(self.step.clone());

            match self.step {
                Step::Solution => {
                    let mut solution: Vec<usize> = self
                        .partial
                        .iter()
                        .map(|&n| self.matrix.row_offsets[n])
                        .collect();
                    solution.sort();
                    return Some(solution);
                }
                Step::Done => break,
                _ => {}
            }
        }
        None
    }

    fn _step(&mut self, step: Step) -> Step {
        match step {
            Step::CoverColumn => self._step_cover_column(),
            Step::CoverRow => self._step_cover_row(),
            Step::UncoverColumn => self._step_uncover_column(),
            Step::UncoverRow => self._step_uncover_row(),
            Step::Solution => Step::UncoverRow,
            Step::Done => Step::Done,
        }
    }

    fn _step_uncover_column(&mut self) -> Step {
        // uncover column
        self.cursors.pop();
        let column = self.columns.pop().expect("This should never happen");

        self.matrix.uncover(column);

        Step::UncoverRow
    }

    fn _step_cover_column(&mut self) -> Step {
        // find shortest column.
        let mut row_cursor = self.matrix.row_links.cursor(H);
        let mut column = match row_cursor.next(&self.matrix.row_links) {
            Some(cell) => cell,
            None => {
                return Step::Solution;
            }
        };
        while let Some(next_column) = row_cursor.next(&self.matrix.row_links) {
            if self.matrix.sizes[next_column.0] < self.matrix.sizes[column.0] {
                column = next_column;
            }
        }

        // cover column
        self.matrix.cover(column);

        // add to state
        self.columns.push(column);
        let cursor = self.matrix.row_links.cursor(column);
        self.cursors.push(cursor);

        Step::CoverRow
    }

    fn _step_cover_row(&mut self) -> Step {
        let cursor = &mut self.cursors.last_mut().expect("No last column.");

        let row_cell = match cursor.next(&self.matrix.column_links) {
            // if run out of row, then step up
            None => {
                return Step::UncoverColumn;
            }
            Some(cell) => cell,
        };

        // cover row
        let mut row_cursor = self.matrix.row_links.cursor(row_cell);
        while let Some(cell) = row_cursor.next(&self.matrix.row_links) {
            self.matrix.cover(self.matrix.heads[cell.0]);
        }

        self.partial.push(row_cell.0);

        // one step down
        Step::CoverColumn
    }

    fn _step_uncover_row(&mut self) -> Step {
        // find last covered row
        let cursor = match self.cursors.last_mut() {
            Some(cursor) => cursor,
            None => {
                return Step::Done;
            }
        };

        // uncover last covered row
        let row_cell = cursor.current;

        let mut row_cursor = self.matrix.row_links.cursor(row_cell);
        while let Some(cell) = row_cursor.prev(&self.matrix.row_links) {
            self.matrix.uncover(self.matrix.heads[cell.0])
        }

        // pop solution
        self.partial.pop();

        Step::CoverRow
    }
}

#[cfg(test)]
pub mod test {
    use super::IterativeSolver;
    use super::Matrix;

    #[test]
    pub fn basics() {
        let mut m = Matrix::new(3);
        m.add_row(&[0, 2]);
    }

    #[test]
    pub fn cover() {
        let mut m = Matrix::new(3);
        m.add_row(&[0, 2]);
        m.add_row(&[1, 2]);
        m.add_row(&[0]);
        m.add_row(&[1]);
        m.add_row(&[2]);

        let solutions = m.solve();
        assert_eq!(solutions.len(), 3);

        assert_eq!(solutions[0], vec![4, 9]);
        assert_eq!(solutions[1], vec![6, 8]);
        assert_eq!(solutions[2], vec![8, 9, 10]);
    }

    #[test]
    pub fn solver() {
        let mut m = Matrix::new(4);
        m.add_row(&[0, 2]);
        m.add_row(&[1, 2, 3]);
        m.add_row(&[0, 1]);
        m.add_row(&[1]);
        m.add_row(&[3]);
        m.add_row(&[2]);
        m.add_row(&[0]);

        let solutions = m.solve();

        let mut solver = IterativeSolver::new(m);

        for n in 0..solutions.len() {
            let solution = solver.next().unwrap();
            assert_eq!(solutions[n], solution);
        }

        solver.next();
    }
}

def is_safe(board, row, col, n):
    for i in range(row):
        if board[i] == col or \
           board[i] - i == col - row or \
           board[i] + i == col + row:
            return False
    return True


def solve_nqueens(n):
    board = [-1] * n

    def solve(row):
        if row == n:
            # ⭐ CHANGED: print chessboard instead of list
            for i in range(n):
                for j in range(n):
                    if board[i] == j:
                        print("Q", end=" ")
                    else:
                        print(".", end=" ")
                print()
            print()
            return True

        for col in range(n):
            if is_safe(board, row, col, n):
                board[row] = col
                if solve(row+1):
                    return True
        return False

    solve(0)


n = int(input("Enter number of queens: "))
solve_nqueens(n)